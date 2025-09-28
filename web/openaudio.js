import { app } from "../../scripts/app.js";

// 添加节点显示名称
app.registerExtension({
    name: "ComfyUI-OpenAudio",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "OpenAudioSaveSpeaker") {
            // 为保存音色节点添加特殊处理
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                // 显示保存结果
                if (message.ui?.text) {
                    const text = message.ui.text.join('\n');
                    this.widgets.forEach(widget => {
                        if (widget.type === "customtext") {
                            widget.value = text;
                        }
                    });
                }
            };
        }
    },
    
    async registerCustomNodes(app) {
        // 注册自定义节点类型
    }
});