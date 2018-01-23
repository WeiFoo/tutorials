from __future__ import print_function,division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt



EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# Plot one example
# print train_data.train_data.size()
# print train_data.train_labels.size()
# print train_data.train_data[0].size()
# plt.imshow(train_data.train_data[0].numpy(), cmap="gray")
# plt.title("%i"%train_data.train_labels[0])
# plt.show()


train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_data = torchvision.datasets.MNIST(root="./mnist",train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1),
                  volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,    #(1, 28, 28)
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),                  #(16, 28, 28)
            nn.MaxPool2d(kernel_size=2) #(16, 14, 14)
        )

        self.conv2 = nn.Sequential(     #(16, 14, 14)
            nn.Conv2d(16,32,5,1,2),     #(32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),            #(32, 7, 7)
        )
        self.out = nn.Linear(32*7*7,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)               #(batch, 32, 7, 7)
        x = x.view(x.size(0), -1)      #(batch, 32*7*7)
        out = self.out(x)
        return out



cnn = CNN()
# print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

# training

for e in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output,1)[1].data.squeeze()
            acc = sum(pred_y == test_y)/test_y.size(0)
            print("Epoch: ", e, "| train loss: %.4f"%loss.data[0],

                  "| test accuracy: %.4f"%acc)

    test_outpt = cnn(test_x[:10])
    pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
    print(pred_y, "prediciton number")
    print(test_y[:10].numpy(), "real number")


