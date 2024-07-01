import torch
import torchaudio
import ChatTTS
from IPython.display import Audio


# 初始化ChatTTS
chat = ChatTTS.Chat()
chat.load_models(source='local',
                 local_path='/mnt/d/VDisk/llm_models/2NoiseChatTTS',
                 compile=False)

texts = [
    """
The driver posted on "Breaking News Commune" that a few days ago, 
he picked up a male middle school student who had a fragrant egg cake but only put 10 yuan into the fare box. 
When the driver reminded him, the student responded with "I don't have any money on me," 
with an indifferent attitude. The original poster thought, 
"You have money to buy an egg cake, but no money to take the bus? This is premeditated crime; 
the bus is not a charity institution."
    """
]

# 生成语音
# wavs = chat.infer(texts, use_decoder=True)
wavs = chat.infer(texts)

# 播放生成的音频
#Audio(wavs[0], rate=24_000, autoplay=True)
torchaudio.save("outputs/read.wav", torch.from_numpy(wavs[0]), 24000)
