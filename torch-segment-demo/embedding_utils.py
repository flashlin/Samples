import torch
import torch.nn as nn


class CharEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        """
        :param input_size: 輸入的字符索引的範圍, 如果使用的字符集包含26个小寫字母（a-z）和4个特殊字符（例如'@'、'#'、'$'、'%'），
        那摩 input_size 應該是 30
        :param embedding_size: GPT-2的 embedding_size為 768
        :param hidden_size: GPT-2的 hidden_size 768
        :param output_size:
        """
        super(CharEmbedding, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)  # 将字符索引转换为嵌入向量
        output, hidden = self.gru(embedded)  # 使用GRU处理嵌入向量序列
        output = self.fc(hidden.squeeze(0))  # 将最后一个隐藏状态映射到输出空间
        return output

