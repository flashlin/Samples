import torch
import torch.nn.functional as F

from network import CharRNN, word_to_chunks, MultiHeadAttention

MAX_WORD_LEN = 5


def test(text):
    s2 = word_to_chunks(text, max_len=MAX_WORD_LEN)
    print(f"{s2}")


test("'$select'")


def chunks_to_tensor(chunks):
    input_data = torch.LongTensor(chunks)
    return input_data


print("")
model = CharRNN(256 + 2, 3, 32)
s2 = word_to_chunks("select")
input = chunks_to_tensor(s2)
print(f"{input=}")
outputs = model(input)
print(f"output {outputs=}")


probabilities = F.softmax(outputs, dim=1)
print(f"{probabilities=}")
predicted_classes = torch.argmax(outputs, dim=1)
print(f"{predicted_classes=}")


v_mean = torch.mean(outputs, dim=0)
predicted_class = torch.argmax(v_mean)
#採用平均
print(f"平均 {predicted_class=}")

predicted_class = torch.argmax(outputs, dim=1)
print(f"直接 {predicted_class=}")


weights = torch.nn.functional.softmax(torch.rand(5), dim=0)  # 假设权重是随机的
final_output = torch.sum(outputs * weights.unsqueeze(-1), dim=0)
print(f"{final_output}")


m1 = MultiHeadAttention(3, 3)
outputs = outputs.unsqueeze(1)  # 這個例子使用的是一個單一的樣本（批量大小為1）。在實際使用時，你可能會一次處理多個樣本，那麼批量大小就大於1了
outputs2 = m1(outputs)
print(f"m {outputs2=}")


def get_probability(output):
    # 应用softmax函数来计算每个类别的概率
    probabilities = torch.nn.functional.softmax(output, dim=1)
    print("Probabilities:", probabilities)

    # 使用argmax来获取最大概率对应的类别
    predicted_class = torch.argmax(probabilities, dim=1)
    print("Predicted class:", predicted_class.item())

n = get_probability(outputs2)
print(f"{n=}")
