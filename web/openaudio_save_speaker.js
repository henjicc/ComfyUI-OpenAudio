import { app } from "../../../scripts/app.js";

// 显示OpenAudio音色保存信息的节点
app.registerExtension({
    name: "ComfyUI-OpenAudio.SaveSpeakerNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "OpenAudioSaveSpeaker") {
            // 添加多行文本显示框
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                // 参考已测试过的代码，检查message.text而不是message.ui?.text
                if (message.text) {
                    // 创建显示区域
                    let widget = this.widgets.find(w => w.name === "save_info");
                    if (!widget) {
                        // 创建新的文本显示widget
                        const container = document.createElement("div");
                        const textBox = document.createElement("div");
                        container.appendChild(textBox);
                        
                        widget = this.addDOMWidget("save_info", "customtext", container, {
                            getValue() {
                                return textBox.textContent || "";
                            },
                            setValue(v) {
                                if (textBox) {
                                    textBox.textContent = v || "";
                                }
                            }
                        });
                        
                        // 设置样式
                        container.style.background = "#222";
                        container.style.color = "#31EC88";
                        container.style.padding = "10px";
                        container.style.borderRadius = "5px";
                        container.style.overflow = "auto";
                        container.style.maxHeight = "150px";
                        
                        textBox.style.whiteSpace = "pre-wrap";
                        textBox.style.fontSize = "12px";
                        textBox.style.wordBreak = "break-word";
                        
                        // 设置widget属性
                        widget.serializeValue = () => {
                            return textBox.textContent || "";
                        };
                    }
                    
                    // 参考已测试过的代码，正确更新文本内容
                    if (widget.element && widget.element.firstChild) {
                        widget.element.firstChild.textContent = message.text[0];
                    } else if (widget.element) {
                        widget.element.textContent = message.text[0];
                    }
                    
                    // 在现有高度基础上增加高度来显示信息，保持用户设置的宽度
                    // 保存当前宽度和高度
                    const currentWidth = this.size[0];
                    const currentHeight = this.size[1];
                    
                    // 获取文本内容的高度
                    const textContent = message.text[0];
                    // 估算文本行数（假设每行大约15像素高）
                    const lines = textContent.split('\n').length;
                    const estimatedHeight = lines * 15 + 10; // 后面的数字是padding和margin的估计值
                    
                    // 在现有高度基础上增加文本所需的高度
                    this.size[1] = currentHeight + estimatedHeight;
                    // 保持用户设置的宽度
                    this.size[0] = currentWidth;
                }
            };
        }
    }
});