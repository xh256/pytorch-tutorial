import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import os


FTRAIN = './data/training.csv'
def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = pd.read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y


class TrainDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.FloatTensor(data)
        self.target = torch.FloatTensor(target)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)
        
        return x, y

    def __len__(self):
        return len(self.data)


class Model(nn.Module):
    def __init__(self, n_features):
         super(Model, self).__init__()
         self.fc1 = nn.Linear(n_features, 100)
         self.fc2 = nn.Linear(100, 30)
         self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x


class ConvNet(nn.Module):
    def __init__(self, in_channels):
        super(ConvNet, self).__init__()
        
        self.act = F.relu
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.pool3 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(11*11*128, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 30)
        
    def forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = self.pool3(self.act(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        m.bias.data.fill_(0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        m.bias.data.fill_(0.0)


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


# Training
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):    
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    print('Epoch {}, Loss {}'.format(epoch, loss.data.numpy()))
    return loss.item()


def evaluate(epoch):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            output = model(data)
            loss = criterion(output, target)
            
        print('Epoch {}, Eval Loss {}'.format(epoch, loss.data.numpy()))
        return loss.item()


X, y = load()#load2d()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

train_dataset = TrainDataset(X_train, y_train)
val_dataset = TrainDataset(X_val, y_val)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=1, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=1, pin_memory=False)

n_features = X_train.shape[1]
model = Model(n_features)
model.apply(weights_init)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)

train_losses = []
val_losses = []
for i in range(1, 400):
    train_loss = train(i)
    val_loss = evaluate(i)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

plt.figure()
plt.plot(np.log(train_losses))
plt.plot(np.log(val_losses))

with torch.no_grad():
    x_test = X[np.random.choice(np.arange(X.shape[0]), size=16), :]
    output = model(torch.from_numpy(x_test))
    y_pred = output.data.numpy()

fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)
