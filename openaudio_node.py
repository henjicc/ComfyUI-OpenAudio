import sys
import os

# 添加项目根目录到Python路径，确保能导入fish_speech模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import queue
import tempfile
import numpy as np
import torch
import soundfile as sf
import folder_paths  # 导入ComfyUI的folder_paths模块
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue, GenerateResponse
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class OpenAudioNode:
    """
    ComfyUI节点用于OpenAudio S1-mini模型推理
    """
    
    # 全局模型实例，避免重复加载
    _model_manager = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OPENAUDIO_MODEL", {
                    "tooltip": "OpenAudio模型对象，通过OpenAudio Model Loader节点获取"
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test of OpenAudio S1-mini model.",
                    "tooltip": "要转换为语音的文本内容"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "控制生成随机性"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": " nucleus sampling参数"
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.1,
                    "min": 0.9,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "重复惩罚因子"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,  # 修正为32位整数范围
                    "tooltip": "随机种子，用于控制生成的随机性"
                }),
            },
            "optional": {
                "reference_audio": ("AUDIO", {
                    "tooltip": "参考音频用于语音克隆（可选）"
                }),
                "reference_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "参考音频对应的文本（可选）"
                }),
                "speaker_data": ("OPENAUDIO_SPEAKER", {
                    "tooltip": "加载的音色数据（可选）"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "OpenAudio"
    DESCRIPTION = "使用OpenAudio S1-mini模型将文本转换为语音"

    def __init__(self):
        pass

    @classmethod
    def _get_model_paths(cls):
        """获取模型路径，使用ComfyUI模型目录"""
        try:
            # 获取节点本身路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 往上一层就是custom_nodes的路径
            custom_nodes_dir = os.path.dirname(current_dir)
            # 再往上一层就是ComfyUI的根目录
            comfyui_root = os.path.dirname(custom_nodes_dir)
            # 在根目录下创建models/OpenAudio目录
            openaudio_models_dir = os.path.join(comfyui_root, "models", "OpenAudio")
            
            # S1-mini模型路径
            llama_checkpoint_path = os.path.join(openaudio_models_dir, "openaudio-s1-mini")
            decoder_checkpoint_path = os.path.join(openaudio_models_dir, "openaudio-s1-mini", "codec.pth")
            decoder_config_name = "modded_dac_vq"
            
            return llama_checkpoint_path, decoder_checkpoint_path, decoder_config_name, openaudio_models_dir
            
        except Exception as e:
            logger.warning(f"无法获取ComfyUI模型目录，使用本地目录: {str(e)}")
            # 回退到本地目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoints_dir = os.path.join(current_dir, "checkpoints")
            
            # S1-mini模型路径
            llama_checkpoint_path = os.path.join(checkpoints_dir, "openaudio-s1-mini")
            decoder_checkpoint_path = os.path.join(checkpoints_dir, "openaudio-s1-mini", "codec.pth")
            decoder_config_name = "modded_dac_vq"
            
            return llama_checkpoint_path, decoder_checkpoint_path, decoder_config_name, checkpoints_dir

    @classmethod
    def _check_and_download_models(cls, llama_checkpoint_path, decoder_checkpoint_path, checkpoints_dir):
        """检查模型是否存在，如果不存在则尝试下载"""
        # 检查模型文件是否存在
        llama_exists = os.path.exists(llama_checkpoint_path)
        decoder_exists = os.path.exists(decoder_checkpoint_path)
        
        if llama_exists and decoder_exists:
            logger.info("模型文件已存在")
            return True
            
        # 如果模型不存在，记录日志并返回False（不自动下载）
        logger.warning("模型文件不存在，请手动下载模型到以下路径:")
        logger.warning(f"  LLaMA模型: {llama_checkpoint_path}")
        logger.warning(f"  Decoder模型: {decoder_checkpoint_path}")
        logger.warning("请从HuggingFace下载模型: https://huggingface.co/fishaudio/openaudio-s1-mini")
        
        return False

    @classmethod
    def _initialize_model(cls, device="cuda"):
        """初始化模型管理器"""
        if cls._model_manager is not None:
            return cls._model_manager
            
        try:
            # 获取模型路径
            llama_checkpoint_path, decoder_checkpoint_path, decoder_config_name, checkpoints_dir = cls._get_model_paths()
            
            # 检查并提示下载模型
            if not cls._check_and_download_models(llama_checkpoint_path, decoder_checkpoint_path, checkpoints_dir):
                raise RuntimeError("模型文件不存在，请先下载模型")
            
            # 检查CUDA可用性
            if not torch.cuda.is_available():
                device = "cpu"
                logger.info("CUDA不可用，使用CPU运行")
            
            precision = torch.half if torch.cuda.is_available() else torch.float32
            
            # 加载LLaMA模型
            llama_queue = launch_thread_safe_queue(
                checkpoint_path=llama_checkpoint_path,
                device=device,
                precision=precision,
                compile=False  # 默认不编译以提高兼容性
            )
            
            # 加载Decoder模型
            decoder_model = load_decoder_model(
                config_name=decoder_config_name,
                checkpoint_path=decoder_checkpoint_path,
                device=device,
            )
            
            # 创建推理引擎
            tts_engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder_model,
                precision=precision,
                compile=False
            )
            
            # 存储模型管理器
            cls._model_manager = {
                "tts_engine": tts_engine,
                "device": device,
                "precision": precision
            }
            
            logger.info("OpenAudio S1-mini模型加载成功")
            return cls._model_manager
            
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise RuntimeError(f"模型初始化失败: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, model, text, temperature, top_p, repetition_penalty, seed, reference_audio=None, reference_text="", speaker_data=None):
        """
        控制节点何时需要重新执行
        当参数变化时（包括随机种子），需要重新执行
        """
        # 创建一个包含所有参数的哈希键
        params = (id(model), text, temperature, top_p, repetition_penalty, seed, id(reference_audio), reference_text, id(speaker_data))
        return hash(params)

    @classmethod
    def generate_speech(cls, model, text, temperature=0.7, top_p=0.8, repetition_penalty=1.1, seed=0,
                       reference_audio=None, reference_text="", speaker_data=None):
        """
        生成语音
        
        Args:
            model (dict): OpenAudio模型对象
            text (str): 覃转换为语音的文本
            temperature (float): 温度参数
            top_p (float): top-p参数
            repetition_penalty (float): 重复惩罚
            seed (int): 随机种子
            reference_audio (optional): 参考音频
            reference_text (str): 参考文本
            speaker_data (optional): 音色数据
            
        Returns:
            tuple: 包含生成音频的元组
        """
        try:
            # 设置随机种子以确保可重现性
            # 限制seed在32位整数范围内
            seed = seed & 0xffffffff
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            
            # 使用传入的模型
            tts_engine = model["tts_engine"]
            
            # 如果提供了音色数据，则使用音色数据中的VQ令牌直接进行推理
            if speaker_data is not None and "audio_tokens" in speaker_data:
                reference_text_from_speaker = speaker_data.get("reference_text", "")
                audio_tokens = speaker_data.get("audio_tokens")
                if reference_text_from_speaker and audio_tokens is not None:
                    logger.info("使用加载的音色数据进行推理")
                    logger.info(f"VQ令牌形状: {audio_tokens.shape}")
                    
                    # 将numpy数组转换为torch张量
                    audio_tokens_tensor = torch.from_numpy(audio_tokens).long().to(tts_engine.decoder_model.device)
                    
                    # 确保形状正确，参考inference.py的处理方式
                    if audio_tokens_tensor.ndim == 2:
                        # 形状应该是 [N, T]，其中N是量化器数量，T是时间步数
                        logger.info(f"VQ令牌张量形状: {audio_tokens_tensor.shape}")
                    else:
                        logger.warning(f"VQ令牌张量形状异常: {audio_tokens_tensor.shape}")
                    
                    # 创建请求对象
                    request = ServeTTSRequest(
                        text=text,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        format="wav",
                        references=[],  # 不使用references，直接使用VQ令牌
                        streaming=False,
                        max_new_tokens=1024,
                        chunk_length=200,
                    )
                    
                    # 手动创建prompt_tokens和prompt_texts
                    prompt_tokens = [audio_tokens_tensor]  # VQ令牌
                    prompt_texts = [reference_text_from_speaker]  # 参考文本
                    
                    # 直接调用LLM推理
                    response_queue = tts_engine.send_Llama_request(request, prompt_tokens, prompt_texts)
                    
                    # 处理结果
                    segments = []
                    while True:
                        wrapped_result = response_queue.get()
                        if wrapped_result.status == "error":
                            raise wrapped_result.response
                        
                        result = wrapped_result.response
                        if not isinstance(result, GenerateResponse):
                            raise TypeError(f"Expected GenerateResponse, got {type(result).__name__}")
                            
                        if result.action != "next":
                            # 解码VQ令牌到音频，使用更宽松的类型检查
                            try:
                                segment = tts_engine.get_audio_segment(result)
                                segments.append(segment)
                            except Exception as e:
                                logger.error(f"VQ解码失败: {str(e)}")
                                raise
                        else:
                            break
                    
                    if not segments:
                        raise RuntimeError("未能生成音频数据")
                    
                    # 合并所有音频片段
                    audio_data = np.concatenate(segments, axis=0)
                    
                    # 获取采样率
                    if hasattr(tts_engine.decoder_model, "spec_transform"):
                        sample_rate = tts_engine.decoder_model.spec_transform.sample_rate
                    else:
                        sample_rate = tts_engine.decoder_model.sample_rate
                    
                    # 确保音频数据是正确的形状 [B, C, T] 格式
                    if audio_data.ndim == 1:
                        # 如果是单声道，形状为[1, 1, T]
                        audio_data = np.expand_dims(np.expand_dims(audio_data, axis=0), axis=0)
                    elif audio_data.ndim == 2:
                        # 如果是[C, T]格式，转换为[1, C, T]
                        audio_data = np.expand_dims(audio_data, axis=0)
                    elif audio_data.ndim > 3:
                        # 如果超过3维，压缩为3维
                        audio_data = audio_data.reshape(1, -1, audio_data.shape[-1])
                    
                    # 返回ComfyUI格式的音频数据
                    audio_output = {
                        "waveform": torch.from_numpy(audio_data).float(),
                        "sample_rate": sample_rate
                    }
                    
                    return (audio_output,)
            
            # 如果没有音色数据，使用正常的推理流程
            # 准备参考音频
            references = []
            
            # 如果提供了参考音频和文本，则使用它们
            if reference_audio is not None and reference_text:
                # 使用上下文管理器确保文件正确关闭
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    temp_filename = tmp_file.name
                    sf.write(temp_filename, reference_audio["waveform"].squeeze().numpy(), 
                            reference_audio["sample_rate"])
                
                try:
                    # 读取临时文件
                    with open(temp_filename, "rb") as f:
                        audio_bytes = f.read()
                    
                    # 创建参考请求
                    references = [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]
                finally:
                    # 确保临时文件被删除
                    try:
                        os.unlink(temp_filename)
                    except Exception as e:
                        logger.warning(f"无法删除临时文件 {temp_filename}: {str(e)}")
            
            # 创建TTS请求
            request = ServeTTSRequest(
                text=text,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                format="wav",
                references=references,
                streaming=False,
                max_new_tokens=1024,
                chunk_length=200,
            )
            
            # 执行推理
            results = list(tts_engine.inference(request))
            
            # 处理结果
            audio_data = None
            sample_rate = 22050  # 默认采样率
            
            for result in results:
                if result.code == "final" and result.audio is not None:
                    sample_rate, audio_array = result.audio
                    if isinstance(audio_array, np.ndarray):
                        audio_data = audio_array
                    break
            
            if audio_data is None:
                raise RuntimeError("未能生成音频数据")
            
            # 确保音频数据是正确的形状 [B, C, T] 格式
            if audio_data.ndim == 1:
                audio_data = np.expand_dims(np.expand_dims(audio_data, axis=0), axis=0)
            elif audio_data.ndim == 2:
                audio_data = np.expand_dims(audio_data, axis=0)
            elif audio_data.ndim > 3:
                audio_data = audio_data.reshape(1, -1, audio_data.shape[-1])
            
            # 返回ComfyUI格式的音频数据
            audio_output = {
                "waveform": torch.from_numpy(audio_data).float(),
                "sample_rate": sample_rate
            }
            
            return (audio_output,)
            
        except Exception as e:
            logger.error(f"语音生成失败: {str(e)}")
            # 返回静音音频作为后备
            silent_audio = {
                "waveform": torch.zeros(1, 1, 22050).float(),
                "sample_rate": 22050
            }
            return (silent_audio,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "OpenAudio": OpenAudioNode,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAudio": "OpenAudio S1-mini TTS",
}