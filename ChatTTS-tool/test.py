import torch
import ChatTTS
from IPython.display import Audio

# 初始化ChatTTS
chat = ChatTTS.Chat()
chat.load_models()

# 定义要转换为语音的文本
texts = ["你好，欢迎使用ChatTTS！"]

# 生成语音
wavs = chat.infer(texts, use_decoder=True)

# 播放生成的音频
Audio(wavs[0], rate=24_000, autoplay=True)