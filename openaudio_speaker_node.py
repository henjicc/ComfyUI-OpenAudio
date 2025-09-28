import os
import torch
import numpy as np
import soundfile as sf
import tempfile
import folder_paths
from loguru import logger

from fish_speech.utils.schema import ServeReferenceAudio


class OpenAudioSaveSpeakerNode:
    """
    ComfyUI节点用于保存OpenAudio音色特征
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OPENAUDIO_MODEL", {
                    "tooltip": "OpenAudio模型对象，通过OpenAudio Model Loader节点获取"
                }),
                "reference_audio": ("AUDIO", {
                    "tooltip": "参考音频用于提取音色特征"
                }),
                "reference_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "参考音频对应的文本"
                }),
                "speaker_name": ("STRING", {
                    "default": "speaker",
                    "tooltip": "音色文件名称"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("speaker_file",)
    FUNCTION = "save_speaker"
    CATEGORY = "OpenAudio"
    OUTPUT_NODE = True
    DESCRIPTION = "保存OpenAudio音色特征到文件"
    WEB_DIRECTORY = "./web"

    def __init__(self):
        pass

    @classmethod
    def _get_speaker_dir(cls):
        """获取音色文件保存目录"""
        try:
            # 获取节点本身路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 往上一层就是custom_nodes的路径
            custom_nodes_dir = os.path.dirname(current_dir)
            # 再往上一层就是ComfyUI的根目录
            comfyui_root = os.path.dirname(custom_nodes_dir)
            # 在根目录下创建models/OpenAudio/speakers目录
            speaker_dir = os.path.join(comfyui_root, "models", "OpenAudio", "speakers")
            
            # 确保目录存在
            os.makedirs(speaker_dir, exist_ok=True)
            
            return speaker_dir
            
        except Exception as e:
            logger.warning(f"无法获取ComfyUI模型目录，使用本地目录: {str(e)}")
            # 回退到本地目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            speaker_dir = os.path.join(current_dir, "speakers")
            os.makedirs(speaker_dir, exist_ok=True)
            
            return speaker_dir

    @classmethod
    def save_speaker(cls, model, reference_audio, reference_text, speaker_name, prompt=None, extra_pnginfo=None):
        """
        保存音色特征到文件
        
        Args:
            model: OpenAudio模型对象
            reference_audio: 参考音频
            reference_text: 参考文本
            speaker_name: 音色文件名称
            prompt: 提示信息
            extra_pnginfo: 额外PNG信息
            
        Returns:
            tuple: 包含音色文件路径的元组
        """
        try:
            # 检查输入参数
            if not speaker_name or speaker_name.strip() == "":
                error_info = f"音色保存失败！\n\n错误信息: 音色名称不能为空\n时间: {cls._get_current_time()}"
                return {"ui": {"text": [error_info]}, "result": ("",)}
            
            if not reference_text or reference_text.strip() == "":
                error_info = f"音色保存失败！\n\n错误信息: 参考文本不能为空\n时间: {cls._get_current_time()}"
                return {"ui": {"text": [error_info]}, "result": ("",)}
            
            # 获取音色保存目录
            speaker_dir = cls._get_speaker_dir()
            
            # 使用上下文管理器确保文件正确关闭
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_filename = tmp_file.name
                sf.write(temp_filename, reference_audio["waveform"].squeeze().numpy(), 
                        reference_audio["sample_rate"])
            
            try:
                # 获取VQ编码器模型
                decoder_model = model["tts_engine"].decoder_model
                
                # 读取临时文件
                audio, sr = sf.read(temp_filename)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)  # 转换为单声道
                
                # 重采样到模型采样率
                import torchaudio
                audio_tensor = torch.from_numpy(audio).float()
                if sr != decoder_model.sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=decoder_model.sample_rate)
                    audio_tensor = resampler(audio_tensor)
                
                # 添加批次维度，确保形状正确 [B, 1, T]
                audios = audio_tensor.unsqueeze(0).unsqueeze(0).to(decoder_model.device)
                audio_lengths = torch.tensor([audios.shape[2]], device=decoder_model.device, dtype=torch.long)
                
                logger.info(f"输入音频形状: {audios.shape}")
                
                # VQ编码，使用更宽松的类型检查
                if hasattr(decoder_model, 'encode') and callable(getattr(decoder_model, 'encode')):
                    result = decoder_model.encode(audios, audio_lengths)
                    # 处理返回值，可能是元组也可能是单个张量
                    if isinstance(result, tuple):
                        indices = result[0]
                    else:
                        indices = result
                    
                    # 确保indices是正确的形状，参考inference.py的处理方式
                    if indices.ndim == 3:
                        indices = indices[0]  # 取第一个元素，形状变为 [N, T]
                    
                    logger.info(f"编码后indices形状: {indices.shape}")
                else:
                    raise ValueError(f"模型不支持encode方法: {type(decoder_model)}")
                
                # 创建音色数据
                speaker_data = {
                    "audio_tokens": indices.cpu().numpy(),  # 保存处理后的indices
                    "reference_text": reference_text,
                    "reference_audio": audio_tensor.cpu().numpy(),  # 保存处理后的音频用于重建
                    "sample_rate": decoder_model.sample_rate,  # 保存采样率信息
                }
                
                # 保存音色文件
                speaker_file_path = os.path.join(speaker_dir, f"{speaker_name}.npz")
                np.savez_compressed(speaker_file_path, **speaker_data)
                logger.info(f"音色文件保存到 {speaker_file_path}")
                
                # 准备显示信息
                save_info = f"音色保存成功！\n\n音色名称: {speaker_name}\n文件路径: {speaker_file_path}\n保存时间: {cls._get_current_time()}"
                
                # 返回保存的文件路径和UI信息
                return {"ui": {"text": [save_info]}, "result": (speaker_file_path,)}
                
            finally:
                # 确保临时文件被删除
                try:
                    os.unlink(temp_filename)
                except Exception as e:
                    logger.warning(f"无法删除临时文件 {temp_filename}: {str(e)}")
                    
        except Exception as e:
            # 准备错误显示信息
            error_info = f"音色保存失败！\n\n错误信息: {str(e)}\n时间: {cls._get_current_time()}"
            logger.error(f"音色保存失败: {str(e)}")
            return {"ui": {"text": [error_info]}, "result": ("",)}
    
    @classmethod
    def _get_current_time(cls):
        """获取当前时间字符串"""
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S')


class OpenAudioLoadSpeakerNode:
    """
    ComfyUI节点用于加载OpenAudio音色特征
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取音色文件列表
        speaker_files = cls._get_speaker_files()
        speaker_options = ["无"] + speaker_files if speaker_files else ["无"]
        
        return {
            "required": {
                "speaker_file": (speaker_options, {
                    "default": "无",
                    "tooltip": "要加载的音色文件"
                }),
            }
        }

    RETURN_TYPES = ("OPENAUDIO_SPEAKER", "STRING")
    RETURN_NAMES = ("speaker_data", "speaker_name")
    FUNCTION = "load_speaker"
    CATEGORY = "OpenAudio"
    DESCRIPTION = "加载OpenAudio音色特征文件"

    def __init__(self):
        pass

    @classmethod
    def _get_speaker_dir(cls):
        """获取音色文件目录"""
        try:
            # 获取节点本身路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 往上一层就是custom_nodes的路径
            custom_nodes_dir = os.path.dirname(current_dir)
            # 再往上一层就是ComfyUI的根目录
            comfyui_root = os.path.dirname(custom_nodes_dir)
            # 在根目录下创建models/OpenAudio/speakers目录
            speaker_dir = os.path.join(comfyui_root, "models", "OpenAudio", "speakers")
            
            return speaker_dir
            
        except Exception as e:
            logger.warning(f"无法获取ComfyUI模型目录，使用本地目录: {str(e)}")
            # 回退到本地目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            speaker_dir = os.path.join(current_dir, "speakers")
            
            return speaker_dir

    @classmethod
    def _get_speaker_files(cls):
        """获取音色文件列表"""
        speaker_dir = cls._get_speaker_dir()
        speaker_files = []
        if os.path.exists(speaker_dir):
            speaker_files = [f for f in os.listdir(speaker_dir) if f.endswith(".npz")]
        return speaker_files

    @classmethod
    def load_speaker(cls, speaker_file):
        """
        加载音色特征文件
        
        Args:
            speaker_file: 音色文件名
            
        Returns:
            tuple: 包含音色数据和音色名称的元组
        """
        try:
            # 处理"无"选项
            if speaker_file == "无" or not speaker_file:
                return (None, "")
            
            # 获取音色文件路径
            speaker_dir = cls._get_speaker_dir()
            speaker_file_path = os.path.join(speaker_dir, speaker_file)
            
            # 检查文件是否存在
            if not os.path.exists(speaker_file_path):
                raise FileNotFoundError(f"音色文件不存在: {speaker_file_path}")
            
            # 加载音色数据
            speaker_data = np.load(speaker_file_path, allow_pickle=True)
            
            # 转换为字典格式
            speaker_dict = {
                "audio_tokens": speaker_data["audio_tokens"],
                "reference_text": str(speaker_data["reference_text"]),
                "reference_audio": speaker_data["reference_audio"],
                "sample_rate": int(speaker_data["sample_rate"]) if "sample_rate" in speaker_data else 22050,
            }
            
            # 获取音色名称（去掉扩展名）
            speaker_name = os.path.splitext(speaker_file)[0]
            
            logger.info(f"音色文件加载成功: {speaker_file_path}")
            return (speaker_dict, speaker_name)
            
        except Exception as e:
            logger.error(f"音色加载失败: {str(e)}")
            raise RuntimeError(f"音色加载失败: {str(e)}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "OpenAudioSaveSpeaker": OpenAudioSaveSpeakerNode,
    "OpenAudioLoadSpeaker": OpenAudioLoadSpeakerNode,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAudioSaveSpeaker": "OpenAudio Save Speaker",
    "OpenAudioLoadSpeaker": "OpenAudio Load Speaker",
}