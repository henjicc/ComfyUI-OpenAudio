from loguru import logger
from omegaconf import OmegaConf

from fish_speech.utils.file import AUDIO_EXTENSIONS, list_files, load_filelist

# register eval resolver，避免重复注册
try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    # 如果已经注册了，就忽略错误
    pass
# This file is used to convert the audio files to text files using the Whisper model.
# It's mainly used to generate the training data for the VQ model.

backends = torchaudio.list_audio_backends()

if "ffmpeg" in backends:
    backend = "ffmpeg"
else:
    backend = "soundfile"

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "{extra[rank]} - <level>{message}</level>"
)
logger.configure(extra={"rank": f"RANK: {RANK} / {WORLD_SIZE}"})
logger.remove()
logger.add(sys.stderr, format=logger_format)


@lru_cache(maxsize=1)
def get_model(
    config_name: str = "modded_dac_vq",
    checkpoint_path: str = "checkpoints/openaudio-s1-mini/codec.pth",
    device: str | torch.device = "cuda",
):
    with initialize(version_base="1.3", config_path="../../fish_speech/configs"):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    logger.info(f"Loaded model")
    return model


@torch.inference_mode()
def process_batch(files: list[Path], model) -> float:
    wavs = []
    audio_lengths = []
    new_files = []
    max_length = total_time = 0

    for file in files:
        try:
            wav, sr = torchaudio.load(
                str(file), backend=backend
            )  # Need to install libsox-dev
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            continue

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = torchaudio.functional.resample(
            wav.cuda(), sr, model.spec_transform.sample_rate
        )[0]
        total_time += len(wav) / model.spec_transform.sample_rate
        max_length = max(max_length, len(wav))

        wavs.append(wav)
        audio_lengths.append(len(wav))
        new_files.append(file)

    files = new_files

    # Pad to max length
    for i, wav in enumerate(wavs):
        wavs[i] = torch.nn.functional.pad(wav, (0, max_length - len(wav)), "constant")

    audios = torch.stack(wavs, dim=0)[:, None]
    audio_lengths = torch.tensor(audio_lengths, device=model.device, dtype=torch.long)

    # Calculate lengths
    indices, feature_lengths = model.encode(audios, audio_lengths)

    # Save to disk
    outputs = indices.cpu().numpy()

    for file, length, feature, audio_length in zip(
        files, feature_lengths, outputs, audio_lengths
    ):
        feature = feature[:, :length]

        # (T,)
        with open(file.with_suffix(".npy"), "wb") as f:
            np.save(f, feature)

    return total_time


@click.command()
@click.argument("folder")
@click.option("--num-workers", default=1)
@click.option("--config-name", default="modded_dac_vq")
@click.option(
    "--checkpoint-path",
    default="checkpoints/openaudio-s1-mini/codec.pth",
)
@click.option("--batch-size", default=64)
@click.option("--filelist", default=None, type=Path)
def main(
    folder: str,
    num_workers: int,
    config_name: str,
    checkpoint_path: str,
    batch_size: int,
    filelist: Path,
):
    if num_workers > 1 and WORLD_SIZE != num_workers:
        assert WORLD_SIZE == 1, "You should either use SLURM or this launcher, not both"

        logger.info(f"Spawning {num_workers} workers")

        if torch.cuda.is_available():
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if visible_devices is None:
                visible_devices = list(range(torch.cuda.device_count()))
            else:
                visible_devices = visible_devices.split(",")
        else:
            # Set to empty string to avoid using GPU
            visible_devices = [""]

        processes = []
        for i in range(num_workers):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(visible_devices[i % len(visible_devices)])
            env["SLURM_PROCID"] = str(i)
            env["SLURM_NTASKS"] = str(num_workers)

            processes.append(
                sp.Popen(
                    [sys.executable] + sys.argv.copy(),
                    env=env,
                )
            )

        for p in processes:
            p.wait()

        logger.info(f"All workers finished")
        return

    # This is a worker
    logger.info(f"Starting worker")
    if