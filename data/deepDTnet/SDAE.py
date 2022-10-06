import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import random

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=12)
parser.add_argument('--matrix_dir', type=str, default='./matrix/PPMI_matrix_Chemical.txt')
args = parser.parse_args()

EPOCH = 10
BATCH_SIZE = args.batch_size
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA, ' use cuda')
# num_cuda = 0
# os.environ['CUDA_VISIBLE_DEVICES'] = str(num_cuda) 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
hidden_dim = args.hidden_size

# ====== Random Seed Initialization ====== #
def seed_everything(seed = 3078):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything()

def checkmkdir(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path, exist_ok=True)
        print("path created")

class MyDataset(object):
  def __init__(self,file_name):
    ppmi_df=pd.read_csv(file_name, sep='\t', header=0)
    x=ppmi_df.iloc[:,1:].values
    self.x_train=torch.tensor(x,dtype=torch.float32)

  def __len__(self):
    return self.x_train.shape[0]
   
  def __getitem__(self,idx):
    return self.x_train[idx]

matrix_dir = args.matrix_dir
trainset = MyDataset(matrix_dir)
train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 2
)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        if (hidden_dim == 3):
            self.encoder = nn.Sequential(
            nn.Linear(732, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
            )
            self.decoder = nn.Sequential(
                nn.Linear(3, 12),
                nn.ReLU(),
                nn.Linear(12, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 732),
                nn.Sigmoid(),       # 픽셀당 0과 1 사이로 값을 출력
            )
        elif (hidden_dim == 12):
            self.encoder = nn.Sequential(
            nn.Linear(732, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            )
            self.decoder = nn.Sequential(
                nn.Linear(12, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 732),
                nn.Sigmoid(),       # 픽셀당 0과 1 사이로 값을 출력
            )
        else:
            raise Exception("hidden_dim is not 3 or 12")
        

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
criterion = nn.MSELoss()


def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img


def train(autoencoder, train_loader):
    autoencoder.train()
    avg_loss = 0
    for step, x in enumerate(train_loader):
        
        noisy_x = add_noise(x)  # 입력에 노이즈 더하기
        noisy_x = noisy_x.to(DEVICE)
        y = x.to(DEVICE)

        encoded, decoded = autoencoder(noisy_x)

        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
    return avg_loss / len(train_loader), encoded

embed = []
for epoch in range(1, EPOCH+1):
    loss, embedding = train(autoencoder, train_loader)
    embed.append(embedding.cpu().detach().numpy())
    print("[Epoch {}] loss:{}".format(epoch, loss))
    
embed = np.concatenate(embed, axis=0)
embeddf = pd.DataFrame(embed)
checkmkdir('./embed')
save_embed_dir = './embed/embed_'+matrix_dir[9:-4]+'_'+str(hidden_dim)+'D.csv'
embeddf.to_csv(save_embed_dir, index=False)