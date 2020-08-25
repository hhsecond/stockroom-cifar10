from stockroom import StockRoom
from stockroom import make_torch_dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def imshow(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



stock = StockRoom()
imgcol = stock.data['cifar10-train-image']
lblcol = stock.data['cifar10-train-label']
# imshow(imgcol[11])


lr = 0.001
momentum = 0.9
check_every = 500
net = Net()
dset = make_torch_dataset([imgcol, lblcol])
dloader = DataLoader(dset, batch_size=64)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)



for epoch in range(2):
    running_loss = 0.0
    current_loss = 99999
    best_loss = 99999
    p = tqdm(dloader)
    p.set_description('[epcoh: %d, iteration: %d] loss: %5d' %(epoch + 1, 1, current_loss))
    for i, data in enumerate(p):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % check_every == check_every - 1:
            current_loss = running_loss / check_every
            running_loss = 0.0
            p.set_description('[epcoh: %d, iteration: %d] loss: %.6f' %(epoch + 1, i + 1, current_loss))
            if current_loss < best_loss:
                with stock.enable_write():
                    stock.experiment['lr'] = lr
                    stock.experiment['momentum'] = momentum
                    stock.model['cifarmodel'] = net.state_dict()
                best_loss = current_loss
print(stock.model.keys())
print('Finished Training')
