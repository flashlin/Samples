import torch
import torchaudio
import ChatTTS
from IPython.display import Audio

# 初始化ChatTTS
chat = ChatTTS.Chat()
chat.load(
    #使用CPU务必删除下一行设备指定cuda
    device="cuda",
    compile = False,
)

# 定义需要转音频的文字内容
texts = ["Hi，大家好 欢迎来到老E的频道，without answer，我的微信公众号是智能生活引擎。专注于互联网、人工智能等技术应用领域，欢迎订阅、关注",]

# 生成音频
wavs = chat.infer(texts, use_decoder=True)

import os
dirs = "output"

if not os.path.exists(dirs):
    os.makedirs(dirs)

# 保存音频文件到本地文件（采样率为24000Hz）
torchaudio.save(".\output\output-01.wav", torch.from_numpy(wavs[0]), 24000)
