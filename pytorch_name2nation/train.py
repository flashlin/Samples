import time

from lstm import device
from utils import randomTrainingExample, save_model, train, categoryFromOutput, timeSince

n_iters = 250000
n_iters = 5000
print_every = 5000
plot_every = 1000

start = time.time()
current_loss = 0
all_losses = []
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    # print(line_tensor)
    output, loss = train(category_tensor.to(device), line_tensor.to(device))
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print(
            '%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


save_model()


import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)
plt.show()



