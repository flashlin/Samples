import torch from torch import nn, optim
import net
CONTEXT_SIZE = 2  # 2-gram
EMBEDDING_DIM = 10  # 词向量的维度

test_sentence = """We always knew our daughter Kendall was
going be a performer of some sort.
She entertained people in our small town
by putting on shows on our front porch when
she was only three or four. Blonde-haired,
blue-eyed, and beautiful, she sang like a
little angel and mesmerized1 everyone.""".split()

trigram = [
    ((test_sentence[i], test_sentence[i+1]), test_sentence[i+2])
    for i in range(len(test_sentence)-2)
]

# 建立每个词与数字的编码，据此构建词嵌入
vocb = set(test_sentence)  # 使用 set 将重复的元素去掉
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}
model = net.n_gram(len(word_to_idx), CONTEXT_SIZE, EMBEDDING_DIM)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-5)
for epoch in range(100):
    train_loss = 0
    for word, label in trigram:
        word = torch.LongTensor([word_to_idx[i] for i in word])  # 将两个词作为输入
        label = torch.LongTensor([word_to_idx[label]])
        # 前向传播
        out = model(word)
        loss = criterion(out, label)
        train_loss += loss.item()
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        print('epoch: {}, Loss: {:.6f}'.format(epoch + 1, train_loss / len(trigram)))

model = model.eval()
word, label = trigram[15]
print('input:{}'.format(word))
print('label:{}'.format(label))
word = torch.LongTensor([word_to_idx[i] for i in word])
out = model(word)
pred_label_idx = out.max(1)[1].item()  # 第一行的最大值的下标
predict_word = idx_to_word[pred_label_idx]  # 得到对应的单词
print('real word is {}, predicted word is {}'.format(label, predict_word)