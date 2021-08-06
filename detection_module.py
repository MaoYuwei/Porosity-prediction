import torch
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from gen_data import *

import numpy as np
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        super(MyDataset, self).__init__()
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform

    def __getitem__(self, index):
        img = self.features[index]
        label = self.labels[index]
        # print(img.size())
        img = self.transform(img)
        # print(img.size())
        return (img, label)

    def __len__(self):
        return self.features.shape[0]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(3 * 3 * 32, 1)

    def forward(self, x):
        # print(x.size())
        out = self.layer1(x)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = out.reshape(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out

def train():
    model.train()
    total_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # images = images.cuda()
        # labels = labels.cuda()
        # Forward pass
        outputs = model(images)

        # print(images.type())
        # print(outputs.type())
        # print(labels.type())
        labels = labels.view(labels.size(0), 1)

        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * outputs.size()[0]

    return total_loss / train_features.shape[0]

def evaluate(loader, show_results=False):
    model.eval()

    pred_list = None
    true_list = None
    total_mse = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):

            images = images.to(device)
            labels = labels.to(device)
            # images = images.cuda()
            # labels = labels.cuda()
            labels = labels.view(labels.size(0), 1)
            pred = model(images)
            # print(images.type())
            # print(pred.type())
            # print(label.type())
            loss = criterion(pred, labels)
            mse = loss.item()
            total_mse += mse * labels.size()[0]

            if show_results == True:
                pred = pred.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                # print(pred.shape)
                # print(label.shape)

                if pred_list is None:
                    pred_list = pred
                    true_list = labels

                else:
                    pred_list = np.append(pred_list, pred)
                    true_list = np.append(true_list, labels)


    return mse, pred_list, true_list

def test_results(layer=0):
    dis = int((subimage_size - 1) / 2)
    data = np.load(filename)[layer]
    # print(data.shape)

    image_list = []
    label_list = []

    # add the before layer
    if layer > 0:
        before_data = np.load(filename)[layer-1]
    else:
        before_data = np.zeros((8, 160, 160))

    for i in range(dis, data.shape[1] - (dis + 1)):
        for j in range(dis, data.shape[2] - (dis + 1)):
            if data[-2, i, j] > 0:  # consider tep feature > 0 is part pixel
                before = before_data[:-2, i - dis:i + dis + 1, j - dis:j + dis + 1]
                image = data[:-2, i - dis:i + dis + 1, j - dis:j + dis + 1]
                image = np.concatenate((before, image))
                label = data[-1, i, j]
                image_list.append(image)
                label_list.append(label)

    test_features = np.array(image_list)
    test_labels = np.array(label_list)

    dataset = MyDataset(features=test_features,
                        labels=test_labels,
                        transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mse, pred, label = evaluate(loader, show_results=True)

    # show image
    dis = int((subimage_size - 1) / 2)
    data = np.load(filename)[layer]

    predict_ret = np.zeros((data.shape[1], data.shape[2]))
    true_ret = np.zeros((data.shape[1], data.shape[2]))

    defect_count = 0
    non_defect_count = 0

    true_defect_count = 0
    true_non_defect_count = 0

    k = 0 # pred index
    for i in range(dis, data.shape[1] - (dis + 1)):
        for j in range(dis, data.shape[2] - (dis + 1)):
            if data[-2, i, j] > 0:  # consider part feature > 0 is part pixel

                # get true image
                l = label[k]
                if l > true_threshold:
                    true_ret[i, j] = 255
                    true_defect_count += 1
                else:
                    true_ret[i, j] = 100
                    true_non_defect_count += 1

                # get predict image
                pred_y = pred[k]
                if pred_y > pred_threshold:
                    predict_ret[i, j] = 255
                    # print(pred_y)
                    defect_count += 1

                else:
                    predict_ret[i, j] = 100
                    non_defect_count += 1

                k += 1

    # print(predict_ret.shape)
    #
    # print(true_ret.shape)
    #
    # print('pred: ', defect_count, non_defect_count)
    # print('true: ', true_defect_count, true_non_defect_count)

    return true_ret, predict_ret


def features_padding(data, out_dim):
    pad_num = out_dim-data.shape[-1]
    before = int(pad_num/2)
    after = pad_num-before
    data = np.pad(data, ((0,0), (0,0), (before, after), (before, after)), 'constant')
    return data

def cal_mean_std(data):
    channel_number = data.shape[1]
    mean_list = []
    std_list = []
    for i in range(channel_number):
        d = data[:, i, :, :]
        mean = np.mean(d)
        std = np.std(d)
        max = np.max(d)
        min = np.min(d)
        # print(mean, std, max, min)
        mean_list.append(mean)
        std_list.append(std)
    return mean_list, std_list

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    subimage_size = 15
    batch_size = 10000
    input_features = 6
    num_epochs = 300
    lr = 1e-5
    input_size = 28
    previous_step = 1
    input_channels = input_features*(previous_step+1)

    mean = [107.12050999054522, 457.5748673340367, 926.6382736731857, 48041.70889020551, 3542.637612071232, 5688.188364275236,
     107.47155333016387, 458.98813576836193, 929.121472228265, 48186.92712630542, 3553.3170538836657, 5699.342931389284]
    std = [78.16336611123124, 259.7409351847671, 2363.0441301142328, 15361.858584770094, 6248.121198850587, 4024.021306394741,
     78.06506670262941, 259.0441940526782, 2365.713105922902, 15158.67120565822, 6252.030356920445, 4016.8431249042383]

    transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])

    criterion = torch.nn.MSELoss()

    model = Net()
    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

    model = torch.load('detection_model.npy')

    true_threshold = 0
    pred_threshold = 0.4

    filename = 'test.npy'
    t = np.load('test.npy')
    l = t.shape[0]

    true_list = np.zeros((l, 160, 160))
    pred_list = np.zeros((l, 160, 160))
    for i in range(l):
        true_ret, pred_ret = test_results(layer=i)
        true_list[i] = true_ret
        pred_list[i] = pred_ret

    np.save(file='test_pred_labels.npy', arr=pred_list)


