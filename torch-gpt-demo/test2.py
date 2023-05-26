import torch
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
print(f'{image=}')


def convert_image_to_tensor(image: list[list[int]]) -> torch.Tensor:
    image_tensor = torch.tensor(image).unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


image_tensor = convert_image_to_tensor(image).unsqueeze(1).type(dtype)
print(f'{image_tensor.shape=}')

layer_output_list, last_state_list = model(image_tensor)

print(f'{layer_output_list=}')
