from tqdm import tqdm
import torch
import config
import torch
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.autograd import Variable


def train_fn(model, data_loader, optimizer):
    model.train()
    fin_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    for i, (image, label) in tqdm(enumerate(data_loader)):
        if torch.cuda.is_available():
            image = Variable(image.cuda())
            label = Variable(label.cuda())
        else:
            image = Variable(image)
            label = Variable(label)

            # forward
        score = model(image)
        loss = criterion(score, label)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent <3
        optimizer.step()

        fin_loss += loss.item()

        if i % 100 == 99:    # print every 100 mini-batches
            print(fin_loss / len(data_loader))
            fin_loss = 0.0

    print('Finished Training')

    return fin_loss / len(data_loader)


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    criterion = torch.nn.CrossEntropyLoss()
    for i, (image, label) in tqdm(enumerate(data_loader)):
        if torch.cuda.is_available():
            image = Variable(image.cuda())
            label = Variable(label.cuda())
        else:
            image = Variable(image)
            label = Variable(label)

        score = model(image)
        loss = criterion(score, label)
        fin_loss += loss.item()
        fin_preds.append(score)

        num_correct = 0
        num_samples = 0

        _, predictions = score.max(1)
        num_correct += (predictions == label).sum()
        num_samples += predictions.size(0)
        print()
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
        print()
    return fin_preds, fin_loss / len(data_loader)
