import os

from id_utils import calculate_checksum, generate_id9, generate_random_id, convert_id_to_numbers
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data import train_loader, tensor2d
import matplotlib.pyplot as plt



def draw_grads(grads_list, param_mapping):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for epoch, gradient in grads_list:
        for item in gradient:
            _, y, value = item
            ax.scatter(epoch, y, value)
    ax.set_yticks(list(param_mapping.values()))
    ax.set_yticklabels(list(param_mapping.keys()))
    return fig


def generate_id():
    id9 = generate_id9()
    id9_numbers = convert_id_to_numbers(id9)
    check_code = calculate_checksum(id9_numbers)
    id_num = int(''.join(str(num) for num in id9_numbers))
    return id_num, check_code

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(121)
    x_list = []
    y_list = []
    for _ in range(500):
        x, y = generate_id()
        x_list.append(x)
        y_list.append(y)
    x_list, y_list = zip(*sorted(zip(x_list, y_list)))
    ax.plot(x_list, y_list, '-')

    bx = fig.add_subplot(122)
    y_list, x_list = zip(*sorted(zip(y_list, x_list)))
    bx.plot(x_list, y_list, '.')
    plt.show()
