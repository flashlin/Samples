import torch
from torch import nn
import numpy as np
from enum import Enum


from ConvGruModel import ConvGRU
from tsql_tokenizr import tsql_tokenize
from vocabulary_utils import WordVocabulary


def reshape_list(items: list[int], shape: tuple[int, int], fill=0):
    extended_items = items + [fill] * (shape[0] * shape[1] - len(items))
    two_dim_array = np.resize(extended_items, shape)
    return two_dim_array


def convert_text_to_image(text: str, shape: tuple[int, int]) -> list[int]:
    tokens = tsql_tokenize(text)
    words = [token.text for token in tokens]
    vob = WordVocabulary()
    value_list = vob.encode_many_words(words)
    converted_value_list = [item.value if isinstance(item, Enum) else item for item in value_list]
    img_arr = reshape_list(converted_value_list, shape, 0)
    return img_arr


use_gpu = torch.cuda.is_available()
if use_gpu:
    dtype = torch.cuda.FloatTensor  # computation in GPU
else:
    dtype = torch.FloatTensor

height = width = 30
channels = 1
model = ConvGRU(input_size=(height, width),
                input_dim=channels,
                hidden_dim=[32, 64],
                kernel_size=(3, 3),
                num_layers=2,
                dtype=dtype,
                batch_first=True,
                bias=True,
                return_all_layers=False)

image = convert_text_to_image("select id from customer", (30, 30))

def convert_image_to_tensor(image: list[list[int]]) -> torch.Tensor:
    image_tensor = torch.tensor(image).unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


image_tensor = convert_image_to_tensor(image).unsqueeze(1).type(dtype)
print(f'{image_tensor.shape=}')

layer_output_list, last_state_list = model(image_tensor)

print(f'{type(layer_output_list[0])=}')
print(f'{layer_output_list[0].shape=}')

relu = nn.ReLU()
fc1 = nn.Linear(30, 128).type(dtype)
fc2 = nn.Linear(128, 10)
logsoftmax = nn.LogSoftmax()


def flatten_list(nested_list: list[list[int]]) -> list[int]:
    flattened_list = [x for sublist in nested_list for x in sublist]
    return flattened_list


out = flatten_list(layer_output_list)
print(f'{out[0].shape=}')


class ClassificationNet(nn.Module):
    def __init__(self, input_size, n_classes, d_type):
        super(ClassificationNet, self).__init__()
        self.linear = nn.Linear(input_size, input_size // 64).type(d_type)
        self.n_classes = n_classes
        self.input_size = input_size

    def forward(self, x):
        x = self.linear(x)
        # x = torch.softmax(x, dim=1)  # 使用 softmax 函数将输出转换为概率分布
        # x = torch.argmax(x, dim=1)   # 取概率最大值对应的索引作为分类结果

        # out = torch.sigmoid(out)  # 限制 0~1 之間
        # out = out * self.n_classes  # 縮放輸出到 0~n_classes 之間
        return x


b = torch.Size([1, 1, 30, 30])
output_tensor = torch.tensor(b)

a_neuron_count = 900 * 64
n_classes = 100
linear_layer = ClassificationNet(900 * 64, n_classes, dtype)
data = out[0]
print(f'{data.shape=}')

a_flattened = data.view(1, -1)
print(f'{a_flattened.shape=}')

output = linear_layer(a_flattened)

print(f'{output=}')
print(f'{output.size()=}')

