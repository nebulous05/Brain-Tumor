import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import random
import cv2
import sys
from model import CNN, threshold
from data_preprocess import MRI


if __name__ == "__main__":
    mri_dataset = MRI()
    mri_dataset.normalize()

    dataloader = DataLoader(mri_dataset, batch_size=32, shuffle=False)
    model = CNN()
    model.load_state_dict(torch.load('./model_weights.pt'))

    model.eval() # put model in test mode
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

    print(accuracy_score(y_true, threshold(outputs)))