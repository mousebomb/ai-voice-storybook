from flask import Flask, request, send_file, render_template_string
import tempfile
import os
import sys
from pydub import AudioSegment
from tqdm import tqdm
import numpy as np

print(tempfile.gettempdir())

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

app = Flask(__name__)

# 初始化CosyVoice2模型
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

def split_text(text, max_length=100):
    """将文本分割成短句子"""
    if not text or len(text.strip()) == 0:
        return []
    
    sentences = []
    current_sentence = ""
    
    text = text.strip()
    for char in text:
        current_sentence += char
        if char in '。！？；.!?' or len(current_sentence) > max_length:
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            current_sentence = ""
    
    if current_sentence and current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return sentences if sentences else [text[:max_length]]

def merge_audio_files(audio_files):
    """合并多个音频文件"""
    combined = AudioSegment.empty()
    for audio_file in audio_files:
        segment = AudioSegment.from_wav(audio_file)
        combined += segment
    return combined

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>文本转有声书</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 8px; }
            .progress { width: 100%; height: 20px; background: #ddd; border-radius: 10px; margin: 20px 0; }
            .progress-bar { width: 0%; height: 100%; background: #4CAF50; border-radius: 10px; transition: width 0.3s; }
            #status { margin: 10px 0; }
            .file-list { margin: 10px 0; }
            .file-item { display: flex; align-items: center; margin: 5px 0; padding: 5px; background: #fff; border-radius: 4px; }
            .file-progress { flex: 1; margin: 0 10px; }
            .file-status { min-width: 100px; }
            .total-progress { margin-top: 20px; }
            .total-progress-label { margin-bottom: 5px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>文本转有声书</h1>
            <form id="uploadForm">
                <p>请上传TXT文本文件：</p>
                <input type="file" name="files" accept=".txt" multiple required>
                <button type="submit">开始转换</button>
            </form>
            <div class="file-list" id="fileList"></div>
            <div class="total-progress">
                <div class="total-progress-label">总体进度：<span id="totalStatus">准备就绪</span></div>
                <div class="progress">
                    <div class="progress-bar" id="totalProgressBar"></div>
                </div>
            </div>
        </div>
        <script>
            let fileQueue = [];
            let currentProcessingIndex = -1;
            let totalFiles = 0;

            function createFileProgressElement(file, index) {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <div>${file.name}</div>
                    <div class="progress file-progress">
                        <div class="progress-bar" id="progressBar_${index}"></div>
                    </div>
                    <div class="file-status" id="status_${index}">等待处理</div>
                `;
                return fileItem;
            }

            function updateTotalProgress() {
                if (totalFiles === 0) return;
                const completedFiles = fileQueue.filter(f => f.completed).length;
                const currentProgress = currentProcessingIndex >= 0 ? 
                    parseInt(document.getElementById(`progressBar_${currentProcessingIndex}`).style.width) : 0;
                const totalProgress = ((completedFiles * 100) + (currentProgress / 100)) / totalFiles;
                document.getElementById('totalProgressBar').style.width = `${totalProgress}%`;
                document.getElementById('totalStatus').textContent = 
                    `处理中 ${completedFiles}/${totalFiles} (${Math.round(totalProgress)}%)`;
            }

            function processNextFile() {
                if (currentProcessingIndex >= 0) {
                    fileQueue[currentProcessingIndex].completed = true;
                }
                currentProcessingIndex++;
                if (currentProcessingIndex >= fileQueue.length) {
                    document.getElementById('totalStatus').textContent = '全部处理完成';
                    return;
                }

                const file = fileQueue[currentProcessingIndex].file;
                const formData = new FormData();
                formData.append('file', file);

                document.getElementById(`status_${currentProcessingIndex}`).textContent = '处理中...';

                fetch('/synthesize', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) throw new Error('上传失败');
                    const contentDisposition = response.headers.get('Content-Disposition');
                    let filename = '';
                    const filenameMatch = /filename\*=UTF-8''([^;]+)/.exec(contentDisposition);
                    if (filenameMatch) {
                        filename = decodeURIComponent(filenameMatch[1]);
                    } else {
                        filename = contentDisposition.split('filename=')[1].replace(/"/g, '');
                    }
                    return response.blob().then(blob => ({ blob, filename }));
                })
                .then(({ blob, filename }) => {
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                    document.getElementById(`status_${currentProcessingIndex}`).textContent = '完成';
                    processNextFile();
                })
                .catch(error => {
                    document.getElementById(`status_${currentProcessingIndex}`).textContent = 
                        `处理失败: ${error.message}`;
                    processNextFile();
                });

                checkProgress();
            }

            function checkProgress() {
                if (currentProcessingIndex >= 0 && currentProcessingIndex < fileQueue.length) {
                    fetch('/progress')
                        .then(response => response.json())
                        .then(data => {
                            const progressBar = document.getElementById(
                                `progressBar_${currentProcessingIndex}`);
                            if (progressBar) {
                                progressBar.style.width = data.progress + '%';
                                document.getElementById(
                                    `status_${currentProcessingIndex}`).textContent = data.status;
                                updateTotalProgress();
                                if (data.progress < 100) {
                                    setTimeout(checkProgress, 1000);
                                }
                            }
                        });
                }
            }

            document.getElementById('uploadForm').onsubmit = function(e) {
                e.preventDefault();
                const files = document.querySelector('input[type=file]').files;
                if (files.length === 0) return;

                fileQueue = Array.from(files).map(file => ({ file, completed: false }));
                totalFiles = files.length;
                currentProcessingIndex = -1;

                const fileList = document.getElementById('fileList');
                fileList.innerHTML = '';
                fileQueue.forEach((item, index) => {
                    fileList.appendChild(createFileProgressElement(item.file, index));
                });

                document.getElementById('totalStatus').textContent = '准备处理...';
                document.getElementById('totalProgressBar').style.width = '0%';

                processNextFile();
            };
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

# 全局变量用于跟踪进度
progress = {"progress": 0, "status": "准备就绪"}

@app.route('/progress')
def get_progress():
    return progress

@app.route('/synthesize', methods=['POST'])
def synthesize():
    global progress
    progress = {"progress": 0, "status": "正在处理文件..."}
    
    if 'file' not in request.files:
        return "请上传文件", 400
    
    file = request.files['file']
    if file.filename == '':
        return "未选择文件", 400
    
    # 读取文本内容
    text = file.read().decode('utf-8')
    
    # 分割文本
    sentences = split_text(text)
    total_sentences = len(sentences)
    print(f"总句子数: {total_sentences}")
    
    # 加载提示语音
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    
    # 临时文件列表
    temp_files = []
    
    try:
        # 处理每个句子
        for i, sentence in enumerate(sentences):
            progress["status"] = f"正在合成第 {i+1}/{total_sentences} 个片段"
            progress["progress"] = int((i / total_sentences) * 90)
            
            # 进行语音合成
            for _, result in enumerate(cosyvoice.inference_zero_shot(sentence, '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    torchaudio.save(temp_file.name, result['tts_speech'], cosyvoice.sample_rate)
                    temp_files.append(temp_file.name)
        
        progress["status"] = "正在合并音频文件..."
        progress["progress"] = 90
        
        # 合并所有音频文件
        combined_audio = merge_audio_files(temp_files)
        
        # 保存最终文件
        output_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        combined_audio.export(output_file.name, format='mp3')
        
        # 在服务器上保存一份副本
        os.makedirs('output', exist_ok=True)
        output_filename = os.path.splitext(file.filename)[0] + '.mp3'
        server_output_path = os.path.join('output', output_filename)
        combined_audio.export(server_output_path, format='mp3')
        
        progress["status"] = "处理完成"
        progress["progress"] = 100
        
        # 使用上传的txt文件名（不含扩展名）作为下载文件名
        download_filename = os.path.splitext(file.filename)[0] + '.mp3'
        return send_file(output_file.name, as_attachment=True, download_name=download_filename)
    
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)