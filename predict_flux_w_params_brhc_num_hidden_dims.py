#%%
import os
os.environ['TORCH_HOME'] = '/sqfs2/cmc/1/work/G15489/v60716/.cache/torch'

# import json
import csv
import math
import multiprocessing
import datetime
from pathlib import Path
# from tqdm import tqdm
import time
import pickle
import ast

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
# import optuna
# from torchsummary import summary
# from sklearn.metrics import accuracy_score

import prediction_model

#%%
# ==========================================================
def result_name(modelname):
    return f'pred_flux_brhc_{modelname}_w_params_num_hidden_dims'
# ==========================================================

params = {
    'batch_size': 128,
    # 'weight_decay': 0.001,
    # 'epochs_optuna': 20,
    'epochs_check': 100,
    'save_every': 10,
    'hidden_dim_init': 4,
    'num_hidden_dims': 4,
    'hidden_dim_other': 8,
    'hidden_dim_other2': 16,
    'learning_rate': 0.001,
    # 'learning_rate': 0.004,
    'hidden_dim_out': 50,
    'hidden_dim_out2': 50,
    'times': 5,
}

#==============================
params_data = {
    # 'ext': 'png',
    'path_data': './_data/data_bmwi3/',
    'folder_image': 'geometry/result/image/',
    'fnames': ['2D', '2U', 'V', 'Nabla'],
    # 'image_size': 256,
    'data_number': 'dataset_number_scaled',
    'data_class': 'dataset_class',
    'data_image': 'dataset_image',
    # 'data_PM': 'dataset_material_PM_dummies',
}
vit_list = [
    'vit_b_16',
    'vit_b_32',
    'vit_l_16',
    'vit_l_32',
    'vit_h_14',
]

#%%
class Regression(nn.Module):
    def __init__(
            self,
            current_dim=2,
            speed_dim=1,
            pm_temp_dim=1,
            pm_material_dim=6,
            hidden_dim_init=6,
            num_hidden_dims=3,
            hidden_dim_out=50,
            hidden_dim_other=25,
            # num_hidden_dims2=5,
            hidden_dim_out2=50,
            hidden_dim_other2=35,
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
        ## init
        self.model_ft = prediction_model.model(params['modelname'], params['typename'], params['pathmodel'])
        num_ftrs = 1000
        self.linear1 = nn.Linear(num_ftrs, hidden_dim_init)
        self.bn1 = nn.BatchNorm1d(hidden_dim_init)
        ## Psi_d
        hidden_dims = [hidden_dim_other]*num_hidden_dims
        hidden_dims[-1] = hidden_dim_out
        linear_list1 = []
        batch_norm_list1 = []
        hidden_dim_before = hidden_dim_init+current_dim+pm_material_dim+pm_temp_dim#+speed_dim
        for hidden_dim in hidden_dims:
            linear_list1.append(nn.Linear(hidden_dim_before, hidden_dim))
            batch_norm_list1.append(nn.BatchNorm1d(hidden_dim))
            hidden_dim_before = hidden_dim
        self.linear_list1 = nn.ModuleList(linear_list1)
        self.batch_norm_list1 = nn.ModuleList(batch_norm_list1)
        self.out1 = nn.Linear(hidden_dim_before, output_dim1)
        ## Psi_q
        hidden_dims2 = [hidden_dim_other2]*num_hidden_dims
        hidden_dims2[-1] = hidden_dim_out2
        linear_list2 = []
        batch_norm_list2 = []
        hidden_dim_before = hidden_dim_init+current_dim+pm_material_dim+pm_temp_dim#+speed_dim
        for hidden_dim in hidden_dims2:
            linear_list2.append(nn.Linear(hidden_dim_before, hidden_dim))
            batch_norm_list2.append(nn.BatchNorm1d(hidden_dim))
            hidden_dim_before = hidden_dim
        self.linear_list2 = nn.ModuleList(linear_list2)
        self.batch_norm_list2 = nn.ModuleList(batch_norm_list2)
        self.out2 = nn.Linear(hidden_dim_before, output_dim2)
        ## activation
        if activation_type=='ReLU':
            self.activation = F.relu
        elif activation_type=='ELU':
            self.activation = F.elu
        else:
            raise NotImplementedError(f'activation type "{activation_type}" is unknown')

    def forward(self, image, parameters, pm_material): # current: 2-dim
        ## image
        x = self.model_ft(image)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = torch.cat([x,parameters], axis=1)
        # x = torch.cat([x,speed], axis=1)
        # x = torch.cat([x,pm_temp], axis=1)
        x = torch.cat([x,pm_material], axis=1)
        for i, (f, bn) in enumerate(zip(self.linear_list1, self.batch_norm_list1)):
            x1 = f(x1) if i > 0 else f(x)
            x1 = bn(x1)
            x1 = self.activation(x1)
        for i, (f, bn) in enumerate(zip(self.linear_list2, self.batch_norm_list2)):
            x2 = f(x2) if i > 0 else f(x)
            x2 = bn(x2)
            x2 = self.activation(x2)
        y1 = self.out1(x1)
        y2 = self.out2(x2)
        return y1, y2 # Psi_d, Psi_q

class ImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        path_data = Path(params_data['path_data'])
        folder_image = params_data['folder_image']
        fnames = params_data['fnames']
        data_number = params_data['data_number']
        data_image = params_data['data_image']
        # data_PM = params_data['data_PM']
        ## image
        image_size = params_data['image_size']
        self.paths = []
        for fname in fnames:
            df_image = pd.read_csv(path_data/f"{data_image}_{fname}.csv")
            path = [path_data/fname/folder_image/p for p in df_image.values.flatten()]
            self.paths.extend(path)
        # self.paths = [p for fname in fnames for p in folder_image.glob(f'{fname}/images/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder_image} for training'
        # num_channels = 3
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        ## label_num
        data_info = {
            'current': ['id','iq'],
            'pm_temp': 'PM_TEMP',
            'flux': ['Psi_d','Psi_q'],
            'pm_material': ['Coercivity','Remanence', 'Recoil', 'Drooping', 'Radius', 'PM_RESISTANCE'],
        }
        self.labels = {}
        for key in data_info.keys():
            self.labels[key] = pd.DataFrame()
        for fname in fnames:
            df = pd.read_csv(path_data/f'{data_number}_{fname}.csv')
            for key, val in data_info.items():
                self.labels[key] = pd.concat((self.labels[key], df[val]), axis=0)
        ## label_pm_material
        # key = 'pm_material'
        # self.labels[key] = pd.DataFrame()
        # for fname in fnames:
        #     self.labels[key] = pd.concat((
        #         self.labels[key], pd.read_csv(path_data/f'{data_PM}_{fname}.csv')
        #     ), axis=0)
        for key in self.labels.keys():
            self.labels[key] = self.labels[key].values
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        current = self.labels['current'][index]
        # speed = self.labels['speed'][index]
        pm_temp = self.labels['pm_temp'][index]
        # x2 = np.concatenate([current, speed, pm_temp])
        x2 = np.concatenate([current, pm_temp])
        pm_material = self.labels['pm_material'][index]
        psi_d, psi_q = self.labels['flux'][index]
        # hysteresis = self.labels['hysteresis'][index]
        # joule = self.labels['joule'][index]

        return (
            self.transform(img),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(pm_material, dtype=torch.float32),
            torch.tensor(psi_d, dtype=torch.float32),
            torch.tensor(psi_q, dtype=torch.float32),
            # torch.tensor(hysteresis, dtype=torch.float32),
            # torch.tensor(joule, dtype=torch.float32),
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
    # print(pred.float(), label.float())
    return nn.MSELoss()(pred.float(), label.float())
def train_step(x1, x2, x3, t1, t2, model, optimizer):
    model.train()
    preds = model(x1, x2, x3)
    t1 = t1.unsqueeze(1)
    t2 = t2.unsqueeze(1)
    loss1 = compute_loss(t1, preds[0])
    loss2 = compute_loss(t2, preds[1])
    optimizer.zero_grad()
    loss = loss1 + loss2
    # print(loss)
    loss.backward()
    optimizer.step()
    return (loss1, loss2), preds
def valid_step(x1, x2, x3, t1, t2, model):
    model.eval()
    preds = model(x1, x2, x3)
    t1 = t1.unsqueeze(1)
    t2 = t2.unsqueeze(1)
    loss1 = compute_loss(t1, preds[0])
    loss2 = compute_loss(t2, preds[1])
    return (loss1, loss2), preds

def main(modelname, typename, pathmodel=None, num_hidden_dims_list=None):
    params['modelname'] = modelname
    params['typename'] = typename
    params['pathmodel'] = pathmodel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CORES = multiprocessing.cpu_count()
    world_size = torch.cuda.device_count()
    is_ddp = world_size > 1
    rank = 0

    ## verify the model with best parameters
    base_dir = './'
    results_dir = 'results'
    name = params['result_name']

    dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M')[2:]
    name = dt_now + '_' + name
    base_dir = Path(base_dir)
    (base_dir / results_dir / name).mkdir(parents=True, exist_ok=True)

    def model_name(num, times, n1):
        return str(base_dir / results_dir / name / f'model_{num}_{times}_{n1}.pt')
    def save_model(model, num, times, n1):
        torch.save(model, model_name(num, times, n1))
    def save_result(result, num, times, n1):
        # with open(str(base_dir / results_dir / name / f'result_{num}_{times}.csv'), 'w', encoding='Shift_jis') as f:
        with open(str(base_dir / results_dir / name / f'result_{times}_{n1}.csv'), 'w', encoding='Shift_jis') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(result)
    def save_best_params(best_params):
        with open(str(base_dir / results_dir / name / f'best_params.pkl'), "wb") as tf:
            pickle.dump(best_params,tf)
    def save_params(params):
        with open(str(base_dir / results_dir / name / f'params.pkl'), "wb") as tf:
            pickle.dump(params,tf)

    save_params(params)
    train_loader, valid_loader = set_data_src(NUM_CORES, params['batch_size'], world_size, rank, is_ddp)
    
    print('model start')
    epochs = params['epochs_check']
    save_every = params['save_every']

    for num_hidden_dims in num_hidden_dims_list:
        params['num_hidden_dims'] = num_hidden_dims
        for t in range(params['times']):
            model = Regression(**params).to(device)
            optimizer = optimizers.Adam(model.parameters(), lr=params['learning_rate'])#, weight_decay=study.best_params['weight_decay'])
            # save_best_params(study.best_params)

            print(f'{t}-times')
            np.random.seed(t+1)
            torch.manual_seed(t+1)
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
                    # train_loss2,
                    valid_loss1,
                    # valid_loss2,
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
                if (epoch+1) % save_every == 0: save_model(model.state_dict(), epoch+1, t, num_hidden_dims)
            save_result(results, epoch+1, t, num_hidden_dims)

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
        "num_hidden_dims_list",
        default=None,
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--pathmodel",
        default=None,
        help='path to nn model'
    )
    args = parser.parse_args()

    params_data['image_size'] = 224 if args.modelname in vit_list else 256
    params['result_name'] = result_name(args.modelname)
    print(params['result_name'])
    print(type(args.num_hidden_dims_list), args.num_hidden_dims_list)
    main(args.modelname, args.typename, args.pathmodel, args.num_hidden_dims_list)


