import os
import torch
import subprocess
import folder_paths
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue


class OpenAudioModelNode:
    """
    ComfyUI节点用于加载和管理OpenAudio S1-mini模型
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["openaudio-s1-mini"], {
                    "default": "openaudio-s1-mini",
                    "tooltip": "要加载的OpenAudio模型名称"
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "模型运行设备"
                }),
                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否自动下载模型"
                }),
            }
        }

    RETURN_TYPES = ("OPENAUDIO_MODEL",)
    RETURN_NAMES = ("openaudio_model",)
    FUNCTION = "load_model"
    CATEGORY = "OpenAudio"
    DESCRIPTION = "加载OpenAudio S1-mini模型用于推理"
    
    def __init__(self):
        pass

    @classmethod
    def _get_model_paths(cls, model_name):
        """获取模型路径，优先使用ComfyUI模型目录"""
        try:
            # 获取ComfyUI的模型目录，创建OpenAudio专用文件夹
            comfyui_models_dir = folder_paths.get_folder_paths("custom_nodes")[0]
            comfyui_root = os.path.dirname(os.path.dirname(comfyui_models_dir))
            openaudio_models_dir = os.path.join(comfyui_root, "models", "OpenAudio")
            
            # 确保OpenAudio模型目录存在
            os.makedirs(openaudio_models_dir, exist_ok=True)
            
            # S1-mini模型路径
            llama_checkpoint_path = os.path.join(openaudio_models_dir, model_name)
            decoder_checkpoint_path = os.path.join(openaudio_models_dir, model_name, "codec.pth")
            decoder_config_name = "modded_dac_vq"
            
            return llama_checkpoint_path, decoder_checkpoint_path, decoder_config_name, openaudio_models_dir
            
        except Exception as e:
            logger.warning(f"无法获取ComfyUI模型目录，使用本地目录: {str(e)}")
            # 回退到本地目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoints_dir = os.path.join(current_dir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            # S1-mini模型路径
            llama_checkpoint_path = os.path.join(checkpoints_dir, model_name)
            decoder_checkpoint_path = os.path.join(checkpoints_dir, model_name, "codec.pth")
            decoder_config_name = "modded_dac_vq"
            
            return llama_checkpoint_path, decoder_checkpoint_path, decoder_config_name, checkpoints_dir

    @classmethod
    def _check_model_files(cls, llama_checkpoint_path, decoder_checkpoint_path):
        """检查模型文件是否存在"""
        llama_exists = os.path.exists(llama_checkpoint_path)
        decoder_exists = os.path.exists(decoder_checkpoint_path)
        
        return llama_exists and decoder_exists

    @classmethod
    def _download_model(cls, model_name, checkpoints_dir):
        """使用git clone自动下载模型"""
        try:
            logger.info(f"正在下载模型 {model_name} 到 {checkpoints_dir}")
            
            # 确保目录存在
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            # 模型下载目标路径
            model_target_path = os.path.join(checkpoints_dir, model_name)
            
            # 如果目录已存在，先删除
            if os.path.exists(model_target_path):
                import shutil
                shutil.rmtree(model_target_path)
            
            # 使用git clone下载模型
            # OpenAudio S1-mini模型的HuggingFace仓库地址
            repo_url = "https://huggingface.co/fishaudio/openaudio-s1-mini"
            
            cmd = [
                "git",
                "clone",
                repo_url,
                model_target_path
            ]
            
            # 执行下载命令
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=checkpoints_dir)
            
            if result.returncode == 0:
                logger.info(f"模型 {model_name} 下载成功")
                return True
            else:
                logger.error(f"模型下载失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"模型下载过程中出错: {str(e)}")
            return False

    @classmethod
    def load_model(cls, model_name, device="auto", auto_download=True):
        """
        加载OpenAudio模型
        
        Args:
            model_name (str): 模型名称
            device (str): 运行设备
            auto_download (bool): 是否自动下载模型
            
        Returns:
            tuple: 包含模型对象的元组
        """
        try:
            # 确定设备
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA不可用，回退到CPU")
                device = "cpu"
            
            logger.info(f"正在加载模型 {model_name} 到设备 {device}")
            
            # 获取模型路径
            llama_checkpoint_path, decoder_checkpoint_path, decoder_config_name, checkpoints_dir = cls._get_model_paths(model_name)
            
            # 检查模型文件
            if not cls._check_model_files(llama_checkpoint_path, decoder_checkpoint_path):
                if auto_download:
                    logger.info("模型文件不存在，尝试自动下载...")
                    if not cls._download_model(model_name, checkpoints_dir):
                        raise RuntimeError(f"模型 {model_name} 下载失败")
                else:
                    # 如果模型不存在且不自动下载，记录日志并抛出异常
                    logger.warning("模型文件不存在，请手动下载模型到以下路径:")
                    logger.warning(f"  LLaMA模型: {llama_checkpoint_path}")
                    logger.warning(f"  Decoder模型: {decoder_checkpoint_path}")
                    logger.warning("请从HuggingFace下载模型: https://huggingface.co/fishaudio/openaudio-s1-mini")
                    raise RuntimeError(f"模型文件不存在: {model_name}")
            
            # 确定精度
            precision = torch.half if device == "cuda" else torch.float32
            
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
            
            # 创建模型对象
            model_obj = {
                "tts_engine": tts_engine,
                "device": device,
                "precision": precision,
                "model_name": model_name
            }
            
            logger.info(f"OpenAudio {model_name} 模型加载成功")
            return (model_obj,)
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise RuntimeError(f"模型加载失败: {str(e)}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "OpenAudioModelLoader": OpenAudioModelNode,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAudioModelLoader": "OpenAudio Model Loader",
}