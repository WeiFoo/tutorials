import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim


LR = 0.01
BATCH_SIZE = 32
EPOCH = 12


x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.1*torch.rand(x.size())

torch_dataset = Data.TensorDataset(
    data_tensor=x,
    target_tensor=y
)

loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

## default network
class Net(torch.nn.Module):
    def __init__(self, n_features=1, n_hidden=20, n_output=1):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

opt_SGD = optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop = optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9,0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
loss_his = [[],[],[],[]]

for epoch in range(EPOCH):
    print epoch
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        for net, opt, l_his in zip(nets, optimizers, loss_his):
            output = net(b_x)
            loss = loss_func(output,b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data[0])

labels = ["SGD", "Momentum", "RMSprop", "Adam"]
for i, his in enumerate(loss_his):
    plt.plot(his, label=labels[i])

plt.legend(loc="best")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.ylim((0,0.2))
plt.show()








