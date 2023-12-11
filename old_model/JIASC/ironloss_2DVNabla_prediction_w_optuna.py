# import os
# import json
import csv
import math
import multiprocessing
import datetime
from pathlib import Path
# from tqdm import tqdm
import time
import pickle

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optimizers
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
# import torch.multiprocessing as mp
# import torchvision
from torchvision import models, transforms
import optuna
# from torchsummary import summary
# from sklearn.metrics import accuracy_score

import prediction_model

params_data = {
    'ext': 'jpg',
    'folder_image': './50_data/ripple/',
    'folder_label': './50_data/ripple_3topol/',
    'fnames': ['2D', 'V', 'nabla'],
    # 'image_size': 256,
    'data_current': 'idiq_all_scaled_train.csv',
    'data_speed': 'speed_all_scaled_train.csv',
    # 'data_ripple': 'torque_ripple_all_scaled_train.csv',
    'data_joule': 'joule_all_scaled_train.csv',
    'data_hysteresis': 'hysteresis_all_scaled_train.csv',
    'Ia_max': 134*3**0.5,
}

params = {
    'batch_size': 128,
    # 'weight_decay': 0.001,
    'epochs_optuna': 20,
    'epochs_check': 100,
    'save_every': 20,
    # 'result_name': 'ironloss_2DVNabla_poolformer_s12_optuna',
    # 'modelname': 'resnet18',
    # 'typename': 'transfer_learning',
    # 'path_model': None,#'./poolformer_s12.pth.tar',
}

vit_list = [
    'vit_b_16',
    'vit_b_32',
    'vit_l_16',
    'vit_l_32',
    'vit_h_14',
]

class Regression(nn.Module):
    def __init__(
            self,
            current_dim=2,
            speed_dim=1,
            # hidden_dims=[50,50],
            cat_layer=1,
            num_hidden_dims=5,
            hidden_dim1=6,
            hidden_dim2=50,
            # hidden_dim_cat=6,
            hidden_dim_other=10,
            output_dim1=1,
            output_dim2=1,
            # weight_id=1.0,
            # weight_iq=1.0,
            # weight_speed=1.0,
            # learning_type='transfer_learning',
            activation_type='ReLU',
            **kwargs,
        ):
        super().__init__()
        ## image
        self.model_ft = prediction_model.model(params['modelname'], params['typename'], params['pathmodel'])
        num_ftrs = 1000
        hidden_dims = [hidden_dim_other]*num_hidden_dims
        hidden_dims[0] = hidden_dim1
        hidden_dims[-1] = hidden_dim2
        # hidden_dims[cat_layer] = hidden_dim_cat
        self.cat_layer = cat_layer
        linear_list = []
        batch_norm_list = []
        assert num_hidden_dims > cat_layer, f'cat_layer {cat_layer} should be less than num of hidden dims {len(hidden_dims)} !'
        hidden_dim_before = num_ftrs
        for i, hidden_dim in enumerate(hidden_dims):
            if i == self.cat_layer:
                hidden_dim_before += current_dim+speed_dim
            linear_list.append(nn.Linear(hidden_dim_before, hidden_dim))
            batch_norm_list.append(nn.BatchNorm1d(hidden_dim))
            hidden_dim_before = hidden_dim
        self.linear_list = nn.ModuleList(linear_list)
        self.batch_norm_list = nn.ModuleList(batch_norm_list)
        self.out1 = nn.Linear(hidden_dim_before, output_dim1)
        self.out2 = nn.Linear(hidden_dim_before, output_dim2)

        if activation_type=='ReLU':
            self.activation = F.relu
        elif activation_type=='ELU':
            self.activation = F.elu
        else:
            raise NotImplementedError(f'activation type "{activation_type}" is unknown')

        # self.weights = torch.tensor([
        #     weight_id,
        #     weight_iq,
        #     weight_speed,
        # ]).to(device)

    def forward(self, image, current, speed): # current: 2-dim
        ## image
        x = self.model_ft(image)
        for i, (f, bn) in enumerate(zip(self.linear_list, self.batch_norm_list)):
            if i==self.cat_layer:
                x = torch.cat([x,current], axis=1)
                x = torch.cat([x,speed], axis=1)
            x = f(x)
            x = bn(x)
            x = self.activation(x)
        y1 = self.out1(x)
        y2 = self.out2(x)
        return y1, y2

class ImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        ## image
        ext = params_data['ext']
        folder_image = Path(params_data['folder_image'])
        folder_label = Path(params_data['folder_label'])
        image_size = params_data['image_size'] if torch.cuda.is_available() else 64
        fnames = params_data['fnames']
        self.paths = [p for fname in fnames for p in folder_image.glob(f'{fname}/images/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder_image} for training'
        # num_channels = 3
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        ## label
        data_names = [
            params_data['data_current'],
            params_data['data_speed'],
            params_data['data_joule'],
            params_data['data_hysteresis'],
        ]
        data_cols = [
            ['id','iq'],
            ['N'],
            ['joule'],
            ['hysteresis'],
        ]
        self.labels = []
        for data_name, data_col in zip(data_names, data_cols):
            df = pd.DataFrame()
            for fname in fnames:
                df = pd.concat([df,
                                pd.read_csv(folder_label / fname / data_name,
                                            index_col=0)])
            df.index = range(df.shape[0])
            self.labels.append(df[data_col])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        current = self.labels[0].iloc[index].values
        speed = self.labels[1].iloc[index].values
        joule = self.labels[2].iloc[index].values
        hysteresis = self.labels[3].iloc[index].values

        return (
            self.transform(img),
            torch.tensor(current, dtype=torch.float32),
            torch.tensor(speed, dtype=torch.float32),
            torch.tensor(joule, dtype=torch.float32),
            torch.tensor(hysteresis, dtype=torch.float32),
        )

def set_data_src(NUM_CORES, batch_size, world_size, rank, is_ddp):
    dataset = ImageDataset()
    n_samples = len(dataset)
    train_size = int(n_samples*0.8)
    valid_size = n_samples - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_sampler = DistributedSampler(
        train_dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True
        ) if is_ddp else None
    train_dataloader = DataLoader(
        train_dataset,
        num_workers = math.ceil(NUM_CORES / world_size),
        batch_size = math.ceil(batch_size / world_size),
        sampler = train_sampler,
        shuffle = not is_ddp,
        drop_last = True,
        pin_memory = True
        )
    valid_sampler = DistributedSampler(
        valid_dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True
        ) if is_ddp else None
    valid_dataloader = DataLoader(
        valid_dataset,
        num_workers = math.ceil(NUM_CORES / world_size),
        batch_size = math.ceil(batch_size / world_size),
        sampler = valid_sampler,
        shuffle = not is_ddp,
        drop_last = True,
        pin_memory = True
        )
    return train_dataloader, valid_dataloader

def get_optimizer(trial, model, optimizer_name='Adam'):
    if optimizer_name=='Adam':
        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        # weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
        optimizer = optimizers.Adam(model.parameters(), lr=lr)#, weight_decay=weight_decay)
    elif optimizer_name=='MomentumSGD':
        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
        optimizer = optimizers.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name=='rmsprop':
        optimizer = optimizers.RMSprop(model.parameters())
    else:
        NotImplementedError(f'optimizer "{optimizer_name}" is unknown')
    return optimizer

def compute_loss(label, pred):
    return nn.MSELoss()(pred.float(), label.float())
def train_step(x1, x2, x3, t1, t2, model, optimizer):
    model.train()
    preds = model(x1, x2, x3)
    loss1 = compute_loss(t1, preds[0])
    loss2 = compute_loss(t2, preds[1])
    optimizer.zero_grad()
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
    return (loss1, loss2), preds
def valid_step(x1, x2, x3, t1, t2, model):
    model.eval()
    preds = model(x1, x2, x3)
    loss1 = compute_loss(t1, preds[0])
    loss2 = compute_loss(t2, preds[1])
    return (loss1, loss2), preds

def main(modelname, typename, pathmodel=None):
    params['modelname'] = modelname
    params['typename'] = typename
    params['pathmodel'] = pathmodel
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CORES = multiprocessing.cpu_count()
    world_size = torch.cuda.device_count()
    is_ddp = world_size > 1
    rank = 0
    # batch_size = params['batch_size']
    ## optimize parameters

    def objective(trial):
        args = {
            'num_hidden_dims': trial.suggest_int('num_hidden_dims',2,10,1),
            'hidden_dim1': trial.suggest_int('hidden_dim1',2,18,4),
            'hidden_dim2': trial.suggest_int('hidden_dim2',10,100,10),
            'hidden_dim_other': trial.suggest_int('hidden_dim_other',10,100,10),
            # 'learning_type': trial.suggest_categorical('learning_type',['transfer_learning','fine_tuning','normal']),
            # 'activation_type': trial.suggest_categorical('activation_type',['ReLU','ELU']),
            # 'optimizer': trial.suggest_categorical('optimizer',['Adam','MomentumSGD','rmsprop']),
            # 'batch_size': trial.suggest_categorical('batch_size',[64,128,256,512]),
        }

        train_loader, valid_loader = set_data_src(NUM_CORES, params['batch_size'], world_size, rank, is_ddp)
        model = Regression(**args).to(device)
        # optimizer = get_optimizer(trial, model, optimizer_name=args['optimizer'])
        optimizer = get_optimizer(trial, model, optimizer_name='Adam')

        epochs = params['epochs_optuna']
        for epoch in range(epochs):
            valid_loss1 = 0.
            valid_loss2 = 0.
            for (x1, x2, x3, t1, t2) in train_loader:
                x1, x2, x3, t1, t2 = x1.to(device), x2.to(device), x3.to(device), t1.to(device), t2.to(device)
                train_step(x1, x2, x3, t1, t2, model, optimizer)
            if epoch+1 == epochs:
                for (x1, x2, x3, t1, t2) in valid_loader:
                    x1, x2, x3, t1, t2 = x1.to(device), x2.to(device), x3.to(device), t1.to(device), t2.to(device)
                    loss, _ = valid_step(x1, x2, x3, t1, t2, model)
                    valid_loss1 += loss[0].item()
                    valid_loss2 += loss[1].item()
                valid_loss1 /= len(valid_loader)
                valid_loss2 /= len(valid_loader)

        return valid_loss1+valid_loss2

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    ## verify the model with best parameters
    base_dir = './'
    results_dir = 'results'
    name = params['result_name']

    dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M')[2:]
    name = dt_now + '_' + name
    base_dir = Path(base_dir)
    (base_dir / results_dir / name).mkdir(parents=True, exist_ok=True)

    def model_name(num):
        return str(base_dir / results_dir / name / f'model_{num}.pt')
    def save_model(model, num):
        torch.save(model, model_name(num))
    def save_result(result, num):
        with open(str(base_dir / results_dir / name / f'result_{num}.csv'), 'w', encoding='Shift_jis') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(result)
    def save_best_params(best_params):
        with open(str(base_dir / results_dir / name / f'best_params.pkl'), "wb") as tf:
            pickle.dump(best_params,tf)
    def save_params(params):
        with open(str(base_dir / results_dir / name / f'params.pkl'), "wb") as tf:
            pickle.dump(params,tf)

    train_loader, valid_loader = set_data_src(NUM_CORES, params['batch_size'], world_size, rank, is_ddp)
    model = Regression(**study.best_params).to(device)
    # optimizer = optimizers.RMSprop(model.parameters())
    # if study.best_params['optimizer']=='Adam':
    optimizer = optimizers.Adam(model.parameters(), lr=study.best_params['learning_rate'])#, weight_decay=study.best_params['weight_decay'])
    # elif study.best_params['optimizer']=='MomentumSGD':
    #     optimizer = optimizers.SGD(model.parameters(), lr=study.best_params['learning_rate'], weight_decay=study.best_params['weight_decay'])
    # elif study.best_params['optimizer']=='rmsprop':
    #     optimizer = optimizers.RMSprop(model.parameters())
    # else:
    #     NotImplementedError(f'optimizer is unknown')
    save_best_params(study.best_params)
    save_params(params)

    print('best model start')
    epochs = params['epochs_check']
    save_every = params['save_every']
    results = []
    time_start = time.time()
    for epoch in range(epochs):
        train_loss1 = 0.
        train_loss2 = 0.
        valid_loss1 = 0.
        valid_loss2 = 0.
        for (x1, x2, x3, t1, t2) in train_loader:
            x1, x2, x3, t1, t2 = x1.to(device), x2.to(device), x3.to(device), t1.to(device), t2.to(device)
            loss, _ = train_step(x1, x2, x3, t1, t2, model, optimizer)
            train_loss1 += loss[0].item()
            train_loss2 += loss[1].item()
        train_loss1 /= len(train_loader)
        train_loss2 /= len(train_loader)
        for (x1, x2, x3, t1, t2) in valid_loader:
            x1, x2, x3, t1, t2 = x1.to(device), x2.to(device), x3.to(device), t1.to(device), t2.to(device)
            loss, _ = valid_step(x1, x2, x3, t1, t2, model)
            valid_loss1 += loss[0].item()
            valid_loss2 += loss[1].item()
        valid_loss1 /= len(valid_loader)
        valid_loss2 /= len(valid_loader)
        elapsed_time = time.time()-time_start
        print('Epoch: {}, Train rmse: {}, Valid rmse: {}, Elapsed time: {:.1f}sec'.format(
            epoch+1,
            train_loss1,
            train_loss2,
            valid_loss1,
            valid_loss2,
            elapsed_time
        ))
        results.append([
            epoch+1,
            train_loss1,
            train_loss2,
            valid_loss1,
            valid_loss2,
            elapsed_time
        ])
        if (epoch+1) % save_every == 0:
            save_model(model.state_dict(), epoch+1)
    save_result(results, epoch+1)

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modelname",
        default='resnet18',
        help='name of nn model'
    )
    parser.add_argument(
        "typename",
        default='transfer_learning',
        help='type of nn learning'
    )
    parser.add_argument(
        "--pathmodel",
        default=None,
        help='path to nn model'
    )
    args = parser.parse_args()

    params_data['image_size'] = 224 if args.modelname in vit_list else 256
    params['result_name'] = f'ironloss_2DVNabla_{args.modelname}_optuna'
    print(params['result_name'])
    main(args.modelname, args.typename, args.pathmodel)


