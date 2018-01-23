from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

## hyper-parameter
EPOCH = 1
LR = 0.01
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST)

train_loader = Data.DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=2)

test_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=False,
    transform=torchvision.transforms.ToTensor())

test_x = Variable(test_data.test_data,volatile=True).type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.test_labels.numpy().squeeze()[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True) # (batch, time_step, input_size))
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # x()
        out = self.out(r_out[:, -1, :])  # (batch, time_step, input_size)
        return out


rnn = RNN()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28, 28))
        b_y = Variable(y)
        output = rnn(b_x)

        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            acc = sum(pred_y == test_y) / test_y.size
            print("Epoch: ", epoch, "| train loss: %.4f" % loss.data[0],
                  "| test accuracy: %.4f" % acc)

    test_output = rnn(test_x[:10].view(-1,28,28))
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, "prediciton number")
    print(test_y[:10], "real number")
