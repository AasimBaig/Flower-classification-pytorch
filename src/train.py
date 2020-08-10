import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.autograd import Variable

import config
import engine
from dataset import FlowerDataset
from model import CNNModel

from sklearn.model_selection import train_test_split


def run_training():
    df = pd.read_csv(config.CSV_FILE)

    train, test = train_test_split(
        df, test_size=0.2, random_state=config.SEED)

    train_dataset = FlowerDataset(train["image_paths"].tolist(),
                                  train["targets"].tolist(),
                                  transform=[
                                      transforms.Resize(
                                          size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225))
    ])

    test_dataset = FlowerDataset(test["image_paths"].tolist(),
                                 test["targets"].tolist(),
                                 transform=[
        transforms.Resize(
            size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False)

    model = CNNModel()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True)

    # overfitting for 1 batch
    # image, label = next(iter(train_loader))

    # image = Variable(image.cuda())
    # label = Variable(label.cuda())
    # output = model(image)
    # print(image[0].size())
    # summary(model, input_size=(3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

    model.train()
    for epoch in range(config.EPOCHS):
        print(f"Epoch -- [{epoch+1} / {config.EPOCHS}] ")

        train_loss = engine.train_fn(model, train_loader, optimizer)
        # itercount = 0
        print(f"training loss - {train_loss}")
    save_checkpoint(model)

    check_accuracy(test_loader)


def check_accuracy(data_loader):
    # model = CNNModel()
    model = torch.load(
        "/media/aasim/383C03243C02DD2E/Kaggle_Comp/flower-classification/flower_classification.pth")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (image, label) in tqdm(enumerate(data_loader)):
            if torch.cuda.is_available():
                image = Variable(image.cuda())
                label = Variable(label.cuda())
            else:
                image = Variable(image)
                label = Variable(label)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    print(
        f"Got {correct} / {total} with accuracy {float(correct)/float(total)*100:.2f}")


def save_checkpoint(model, filename="my_checkpoint.bin"):
    print("=> Saving checkpoint")
    PATH = './flower_classification.pth'
    torch.save(model, PATH)


if __name__ == "__main__":
    run_training()
