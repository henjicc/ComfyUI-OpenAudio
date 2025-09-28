from typing import Callable

import torch
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC


class VQManager:

    def __init__(self):
        # Make Pylance happy (attribut/method not defined...)
        self.decoder_model: DAC
        self.load_audio: Callable

    def decode_vq_tokens(self, codes):
        feature_lengths = torch.tensor(
            [codes.shape[1]], device=self.decoder_model.device
        )
        logger.info(f"VQ features: {codes.shape}")

        # 更宽松的类型检查
        if hasattr(self.decoder_model, 'decode') and callable(getattr(self.decoder_model, 'decode')):
            # 确保codes的形状正确
            if codes.ndim == 2:
                codes = codes[None]  # 添加批次维度
            
            return self.decoder_model.decode(
                indices=codes,
                feature_lengths=feature_lengths,
            )[0].squeeze()

        raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

    def encode_reference(self, reference_audio, enable_reference_audio):
        if enable_reference_audio and reference_audio is not None:
            # Load audios, and prepare basic info here
            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate
            reference_audio_content = self.load_audio(reference_audio, sample_rate)

            audios = torch.from_numpy(reference_audio_content).to(
                self.decoder_model.device
            )[None, None, :]
            audio_lengths = torch.tensor(
                [audios.shape[2]], device=self.decoder_model.device, dtype=torch.long
            )
            logger.info(
                f"Loaded audio with {audios.shape[2] / sample_rate:.2f} seconds"
            )

            # VQ Encoder
            # 更宽松的类型检查
            if hasattr(self.decoder_model, 'encode') and callable(getattr(self.decoder_model, 'encode')):
                result = self.decoder_model.encode(audios, audio_lengths)
                # 处理返回值，可能是元组也可能是单个张量
                if isinstance(result, tuple):
                    prompt_tokens = result[0]
                else:
                    prompt_tokens = result
                
                # 如果prompt_tokens是3D张量，取第一个元素
                if prompt_tokens.ndim == 3:
                    prompt_tokens = prompt_tokens[0]
                    
                logger.info(f"Encoded prompt: {prompt_tokens.shape}")
            else:
                raise ValueError(f"Unknown model type: {type(self.decoder_model)}")
        else:
            prompt_tokens = None
            logger.info("No reference audio provided")

        return prompt_tokens
