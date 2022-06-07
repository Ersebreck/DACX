import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data.dataset import Dataset
import json
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
import pandas as pd
import os.path as osp
from torch.autograd import Variable
import torch.nn.functional as F

from nets import *
from utils import *
from C_dataloader import *
from custom_loss import *

# Arguments
parser = argparse.ArgumentParser(description='PyTorch Deeplab v3 Example')
parser.add_argument('--batch-size', type=int, default=140, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=11, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--name', type=str, default='Baseline',
                    help='file on which to save model weights')
parser.add_argument('--net', type=str, default='resnext',
                    help='red de clasificacion')
parser.add_argument('--opt', type=str, default='radam',
                    help='optimizador de la red')
parser.add_argument('--criterion', type=str, default='bce',
                    help='optimizador de la red')


args = parser.parse_args()


img_folder = "../../DACX_1/images/"
train_json = "train.json"
test_json = "test.json"

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomRotation(2),
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.5, 1.5),
                            shear=None, resample=False,
                            fillcolor=tuple(np.array(np.array(mean)*255).astype(int).tolist())),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Set
device = torch.device("cuda")
seed = 666
batch_size = args.batch_size
max_epoch_number = args.epochs
learning_rate = 1e-4
num_workers = 16
test_freq = 50
save_freq = 5
save_path = f"models/{args.name}/"

if "custom" in args.criterion:
    criterion = custom_loss() # implementar el custome
if "KLD" in args.criterion:
    criterion = nn.KLDivLoss()
elif "cross_entropy" in args.criterion:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCELoss()
    
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

# Initialize the dataloaders for training.
test_annotations = os.path.join( 'test.json')
train_annotations = os.path.join( 'train.json')

test_dataset = DAXCDataset(img_folder, test_annotations, val_transform)
train_dataset = DAXCDataset(img_folder, train_annotations, train_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

num_train_batches = int(np.ceil(len(train_dataset) / batch_size))

# Initialize the model

if "resnext" in args.net:
    model = Resnext50(len(train_dataset.classes))
elif "eff" in args.net:
    model = eff(len(train_dataset.classes))
elif "convnext" in args.net:
    model = convext(len(train_dataset.classes))
elif "vit" in args.net:
    model = vit_b_16(len(train_dataset.classes))

# Switch model to the training mode
model.train()
model = model.to(device)
load_model = False
if osp.exists(f"models/{args.name}/checkpoint-000010.pth"):
    with open(f"models/{args.name}/checkpoint-000010.pth", 'rb') as fp:
        state = torch.load(fp)
        model.load_state_dict(state)
        load_model = True


if "radam" in args.opt:
    optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
elif "adamw" in args.opt:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
elif "sgd" in args.opt:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
elif "nadam" in args.opt:
    optimizer = torch.optim.Nadam(model.parameters(), lr=learning_rate)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

#make dirs
os.makedirs(save_path, exist_ok=True)

def MLC_train():
    iteration = 0
    for i in tqdm(range(0,max_epoch_number)):
        batch_losses = []
        for imgs, targets in train_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
    
            model_result = model(imgs)
            loss = criterion(model_result, targets.type(torch.float))
    
            batch_loss_value = loss.item()
            loss.backward()
            optimizer.step()
    
            batch_losses.append(batch_loss_value)
    
            if iteration % test_freq == 0:
                model.eval()
                with torch.no_grad():
                    model_result = []
                    targets = []
                    for imgs, batch_targets in test_dataloader:
                        imgs = imgs.to(device)
                        model_batch_result = model(imgs)
                        model_result.extend(model_batch_result.cpu().numpy())
                        targets.extend(batch_targets.cpu().numpy())
    
                result = calculate_metrics(np.array(model_result), np.array(targets))
                print("epoch:{:2d} iter:{:3d} test: "
                      "micro f1: {:.3f} "
                      "macro f1: {:.3f} "
                      "samples f1: {:.3f}".format(i, iteration,
                                                  result['micro/f1'],
                                                  result['macro/f1'],
                                                  result['samples/f1']))
                loss_value = np.mean(batch_losses)
                if iteration==0:
                    df = pd.DataFrame({'iter': [iteration],'loss': [loss_value], "microF1" : [result['micro/f1']], "macroF1" : [result['macro/f1']], "sampleF1" : [result['samples/f1']]})
                else:
                    df1 = pd.DataFrame({'iter': [iteration],'loss': [loss_value], "microF1" : [result['micro/f1']], "macroF1" : [result['macro/f1']], "sampleF1" : [result['samples/f1']]})
                    df = pd.concat([df, df1], ignore_index=True)
    
                    
    
                model.train()
            iteration += 1
    
        loss_value = np.mean(batch_losses)
        print("epoch:{:2d} iter:{:3d} train: loss:{:.3f}".format(i, iteration, loss_value))
        if i % save_freq == 0 or i == max_epoch_number-1:
            checkpoint_save(model, save_path, i)
            df.to_csv(save_path+"train_log.csv", sep=',', header=True)   

def test():
    model.eval()
    with torch.no_grad():
        model_result = []
        targets = []
        batch_losses = []
        iteration = 0
        for imgs, batch_targets in test_dataloader:
            imgs = imgs.to(device)
            model_batch_result = model(imgs)
            model_result.extend(model_batch_result.cpu().numpy())
            targets.extend(batch_targets.cpu().numpy())
    
            result = calculate_metrics(np.array(model_result), np.array(targets))
            print("iter:{:3d} test: "
                      "micro f1: {:.3f} "
                      "macro f1: {:.3f} "
                      "samples f1: {:.3f}".format(iteration,
                                                  result['micro/f1'],
                                                  result['macro/f1'],
                                                  result['samples/f1']))
            if iteration==0:
                df = pd.DataFrame({'iter': [iteration], "microF1" : [result['micro/f1']], "macroF1" : [result['macro/f1']], "sampleF1" : [result['samples/f1']]})
            else:
                df1 = pd.DataFrame({'iter': [iteration], "microF1" : [result['micro/f1']], "macroF1" : [result['macro/f1']], "sampleF1" : [result['samples/f1']]})
                df = pd.concat([df, df1], ignore_index=True)
                iteration+=1
    df.to_csv(save_path+"test_log.csv", sep=',', header=True)
    

def predict_image(image):
    image_tensor = test_transforms(image)
    image_tensor = image_tensor.unsqueeze_(0)
    input_image = Variable(image_tensor)
    input_image = input_image.to(device)
    output = model(input_image)
    index = output.data.cpu().numpy().argmax()
    return index

    
if __name__ == "__main__":
    best_loss = None
    #if load_model:
        #print("old model")
        #best_loss = test()
    
    MLC_train()
    test()



