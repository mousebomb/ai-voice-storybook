## 文本转有声书 （讲故事项目）

我需要开发一个python应用，功能是文本生成有声书。

简单的说就是输入txt，得到mp3。

运行起来后，会在本地http提供一个web界面，让用户上传一个txt文件（可能有几万字或更长），接收到文本之后，需要调用cosyvoice进行音频合成。

注意，cosyvoice的调用需要拆分为短句子之后逐个调度生成。

生成所有拆分片段后，需要合并为最终一个连贯的mp3文件。

参考资料1: 音频合成部分的模型和参数可参考这段代码：

```py
from flask import Flask, request, send_file
import tempfile
import os
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

app = Flask(__name__)

# 初始化CosyVoice2模型
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    # 获取请求中的文本
    text = request.form.get('text')
    if not text:
        return "请提供文本", 400

    # 加载提示语音
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

    # 进行语音合成
    for i, j in enumerate(cosyvoice.inference_zero_shot(text, '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
        # 将合成的语音保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            torchaudio.save(temp_file.name, j['tts_speech'], cosyvoice.sample_rate)
            temp_file_path = temp_file.name

    # 返回合成的语音文件
    return send_file(temp_file_path, as_attachment=True, download_name='synthesized.wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)

```


# 部署方式：
在conda 中创建虚拟环境：cosyvoice，部署好cosyvoice之后，
然后把 app.py放到cosyvoice目录下，然后安装库`pip install Flask`，
运行`python app.py`。
在局域网内使用5000端口访问即可。如 `http://192.168.50.18:5000/`

部署cosyvoice可参考：
- https://github.com/FunAudioLLM/CosyVoice
- https://www.youtube.com/watch?v=hitJGosX7DE

# 更换音色
在cosyvoice目录下的 `asset` 目录下，有一个 `zero_shot_prompt.wav` 文件，这个文件就是音色。自己录制一个wav文件，朗读 “希望你以后能够做的比我还好呦。”，替换这个文件即可。