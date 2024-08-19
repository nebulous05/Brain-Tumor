import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import random
import cv2
import sys
from data_preprocess import MRI

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=1),
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1) # flatten each element in the mini-batch
        x = self.fc_model(x)
        x = torch.sigmoid(x)
        return x

def threshold(scores, threshold=0.5, minimum=0, maximum=1.0):
    x = np.array(list(scores))
    x[x >= threshold] = maximum
    x[x < threshold] = minimum
    return x


if __name__ == "__main__":
    mri_dataset = MRI()
    mri_dataset.normalize()

    device = torch.device('cpu')

    model = CNN()
    dataloader = DataLoader(mri_dataset, batch_size=32, shuffle=False)

    model.eval() # put the model in test mode
    outputs = []
    y_true = []

    with torch.no_grad(): # turn off gradient computations
        for d in dataloader:
            image = d['image']
            label = d['label']

            y_hat = model(image)

            outputs.append(y_hat)
            y_true.append(label)

    outputs = np.concatenate(outputs, axis=0).squeeze()
    y_true = np.concatenate(y_true, axis=0).squeeze()

    eta = 0.0001
    EPOCH = 400
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)
    dataloader = DataLoader(mri_dataset, batch_size=32, shuffle=True)

    model.train() # put the model in train mode

    for epoch in range(1, EPOCH):
        losses = []
        for d in dataloader:
            optimizer.zero_grad() # make sure optimizer doesn't cache anything

            data = d['image']
            label = d['label']

            y_hat = model(data)

            # define loss function
            error = nn.BCELoss()
            loss = torch.sum(error(y_hat.squeeze(), label))

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        if (epoch+1) % 10 == 0:
            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch+1, np.mean(losses)))

    torch.save(model.state_dict(), './model_weights.pt')