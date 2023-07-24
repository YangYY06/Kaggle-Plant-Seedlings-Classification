import sys
import os
import torch
import torchvision
import torch.nn as nn
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets.folder import DatasetFolder
from PIL import Image
from tqdm import tqdm
from torch.optim import Adam, SGD, Adagrad
from torch.nn import CrossEntropyLoss
import numpy as np
import pandas as pd
import argparse

class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        return self.model(x)

class Densenet(nn.Module):
    def __init__(self, num_classes):
        super(Densenet, self).__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.hidden1 = nn.Linear(37632, 1024)
        self.hidden2 = nn.Linear(1024, 1024)
        self.hidden3 = nn.Linear(1024, num_classes)
        self.relu=nn.ReLU()

    def forward(self, x):
        x = self.pool(x)
        x= x.reshape(-1,37632)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        return x


class Dataset(data.Dataset):

    def __init__(self, img_paths, img_labels=None, transform=None):
        self.transform = transform
        self.img_paths = img_paths
        self.img_labels = img_labels

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.img_labels is not None:
            label = self.img_labels[idx]
            return img, label
        else:
            return img, idx

    def __len__(self):
        return len(self.img_paths)


# data augmentation and pre-processing
class trans():
    def __init__(self):
        self.train = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.RandomRotation(degrees=(-8, 8)),
                                         transforms.RandomCrop((224, 224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                saturation=0.2, hue=0.2),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])


        self.val = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.CenterCrop((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])


# training process
def train(num_epochs, model, optimizer, criterion, train_loader, val_loader, batch_size, writer):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0.
        steps = len(train_loader.dataset) // batch_size + 1
        with tqdm(total=steps) as progress_bar:
            for i, (x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
                writer.add_scalar('train_loss', loss.item(), epoch * len(train_loader) + i)
                writer.close()
                progress_bar.update(1)
                correct += torch.sum(torch.argmax(y_pred, dim=-1) == y)
            train_acc = float(correct.item()) / float(len(train_loader.dataset))
            print("Epoch %d: train correct: %.4f" % (epoch, train_acc))
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.close()
        steps = len(val_loader.dataset) // batch_size + 1
        correct = 0.
        model.eval()

        with tqdm(total=steps) as progress_bar:
            for i, (x, y) in enumerate(val_loader):
                x, y = x.cuda(), y.cuda()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                progress_bar.set_postfix(loss=loss.item())
                progress_bar.update(1)

                correct += torch.sum(torch.argmax(y_pred, dim=-1) == y)

            val_acc = float(correct.item()) / float(len(val_loader.dataset))
            print("Epoch %d: val correct: %.4f" % (epoch, val_acc))
            writer.add_scalar('val_acc', val_acc, epoch)
            writer.close()
    torch.save(model.state_dict(), "model.pth")
    return


# prediction for test dataset
def test(model, test_paths, test_loader, classes, filename):
    test_label_indices = []
    test_img_indices = []

    for i, (x, img_idx) in enumerate(test_loader):
        x = x.cuda()
        y_pred = model(x)
        test_label_indices.extend(list(torch.argmax(y_pred, dim=-1).cpu().numpy()))
        test_img_indices.extend(list(img_idx.cpu().numpy()))

    test_names = [test_paths[idx] for idx in test_img_indices]
    test_names = [name.split("\\")[-1] for name in test_names]
    test_labels = [classes[idx] for idx in test_label_indices]
    test_labels = [labels.split("\\")[-1] for labels in test_labels]

    out_df = pd.DataFrame({'file': test_names, 'species': test_labels})
    out_df.to_csv(os.path.join('results',filename) + '.csv', index=False)
    return


def run(args):
    train_path = './train'
    test_paths = glob('./test/*.png')
    batch_size = args.batchsize
    num_workers = 0
    num_epochs = args.num_epochs
    train_paths = []
    val_paths = []
    train_labels = []
    val_labels = []

    class_paths = glob(train_path + '/*')

    classes = [path.split("/")[-1] for path in class_paths]

    for i, cpath in enumerate(class_paths):
        paths = glob(cpath + '/*.png')
        train_split = int(len(paths) * 0.8)

        train_paths.extend(paths[:train_split])
        train_labels.extend([i] * train_split)

        val_paths.extend(paths[train_split:])
        val_labels.extend([i] * (len(paths) - train_split))

    transform = trans()
    train_dataset = Dataset(train_paths, train_labels, transform=transform.train)
    val_dataset = Dataset(val_paths, val_labels, transform=transform.val)
    test_dataset = Dataset(test_paths, transform=transform.val)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              shuffle=False)
    # define model
    if args.model == 'ResNet':
        model = ResNet(num_classes=12)
    if args.model == 'VGG':
        model = VGG(num_classes=12)
    if args.model == 'DenseNet':
        model = Densenet(num_classes=12)
    if args.model == 'MLP':
        model = MLP(num_classes=12)

    # define optimizer
    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.optimizer == 'Adagrad':
        optimizer = Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    criterion = CrossEntropyLoss()
    model = model.cuda()
    filename = args.exp_name
    writer = SummaryWriter(log_dir=os.path.join('log', filename))
    train(num_epochs, model, optimizer, criterion, train_loader, val_loader, batch_size, writer)
    model.load_state_dict(torch.load("model.pth"))
    test(model, test_paths, test_loader, classes, filename)
    return


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-e', type=str, required=True,
                            help="output will be saved in exp_name")
    arg_parser.add_argument('--batchsize', '-b', type=int, default=64,
                            help="batch size")
    arg_parser.add_argument('--num_epochs', '-n', type=int, default=20,
                            help="total epoch number for training")
    arg_parser.add_argument('--lr', '-l', type=float, default=1e-4,
                            help="learning rate")
    arg_parser.add_argument('--model', '-m', type=str, default='ResNet',
                            help="model")
    arg_parser.add_argument('--optimizer', '-o', type=str, default='Adam',
                            help="optimizer")
    arg_parser.add_argument('--weight_decay', '-w', type=float, default=0,
                            help="weight decay")
    args = arg_parser.parse_args()
    run(args)
