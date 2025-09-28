# ComfyUI-OpenAudio

[OpenAudio (Fish Speech)](https://github.com/fishaudio/fish-speech) 的 ComfyUI 节点。除原项目代码外，其余均为 AI 编写，因此可能存在未知错误，使用时请留意。
只做了 OpenAudio S1 mini 模型的适配，以前的 Fishspeech 模型暂时还没有支持。

## 功能特点

- **零样本声音克隆**：通过简短的参考音频，克隆任何人的声音
- **跨语言合成**：支持多种语言间的语音合成（中、英、日、韩等）
- **自动模型下载**：首次使用时自动下载所需模型

## 安装方法


```bash
cd custom_nodes
git clone https://github.com/henjicc/ComfyUI-OpenAudio.git
cd ComfyUI-OpenAudio
pip install -r requirements.txt
```

## 使用指南

启动 ComfyUI 后，可以在`OpenAudio`分类下找到所有节点。

## 多语言支持

OpenAudio S1 mini 支持以下语言：
- 中文
- 英语
- 日语
- 德语
- 法语
- 西班牙语
- 韩语
- 阿拉伯语
- 俄语
- 荷兰语
- 意大利语
- 波兰语
- 葡萄牙语

## 免责声明

本项目提供的内容仅用于学术研究和技术展示目的，旨在推动语音合成技术的发展。请确保在使用本项目生成的语音内容时，遵守相关法律法规，不得用于任何违法违规用途。所有音频的生成和使用应获得相关权利人的明确授权，避免侵犯他人肖像权、声音权等合法权益。开发者不对因不当使用本项目而导致的任何法律后果承担责任。