#%%
# import json
# import csv
import math
# import multiprocessing
# import datetime
from pathlib import Path
from tqdm import tqdm
# import time
import pickle

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

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
base_dir = '..\\'

params_flux = {
    'modelname':'swin_t',
    'typename':'transfer_learning',
    'pathmodel':None,
    'base_dir_model':f'{base_dir}_result\\material\\1_brhc\\2305100546_pred_flux_brhc_swin_t_w_params\\',
    'trained_model': 'model_100_0.pt',
    'batch_size': 128,
    # 'weight_decay': 0.001,
    # 'epochs_optuna': 20,
    'epochs_check': 100,
    'save_every': 100,
    # 'hidden_dim_init': 8,
    # 'num_hidden_dims': 2,
    # 'hidden_dim_out': 50,
    # 'hidden_dim_other': 10,
    # 'num_hidden_dims2': 2,
    # 'hidden_dim_out2': 50,
    # 'hidden_dim_other2': 12,
    'learning_rate': 0.004,
    'times': 1,
}
dir_model = params_flux['base_dir_model']
with open(f'{dir_model}params.pkl', "rb") as tf:
    params_ = pickle.load(tf)
params_flux.update(params_)

params_ironloss = {
    'modelname':'swin_t',
    'typename':'transfer_learning',
    'pathmodel':None,
    'base_dir_model':f'{base_dir}_result\\material\\1_brhc\\2305100546_pred_ironloss_brhc_swin_t_w_params\\',
    'trained_model': 'model_100_0.pt',
    'batch_size': 128,
    # 'weight_decay': 0.001,
    # 'epochs_optuna': 20,
    'epochs_check': 100,
    'save_every': 100,
    # 'hidden_dim_init': 8,
    # 'num_hidden_dims': 2,
    # 'hidden_dim_out': 50,
    # 'hidden_dim_other': 10,
    # 'num_hidden_dims2': 2,
    # 'hidden_dim_out2': 50,
    # 'hidden_dim_other2': 12,
    'learning_rate': 0.004,
    'times': 1,
}

dir_model = params_ironloss['base_dir_model']
with open(f'{dir_model}params.pkl', "rb") as tf:
    params_ = pickle.load(tf)
params_ironloss.update(params_)

params_torque = {
    'modelname':'swin_t',
    'typename':'transfer_learning',
    'pathmodel':None,
    'base_dir_model':f'{base_dir}_result\\material\\1_brhc\\2305101331_pred_torque_brhc_swin_t_w_params\\',
    'trained_model': 'model_100_0.pt',
    'batch_size': 128,
    # 'weight_decay': 0.001,
    # 'epochs_optuna': 20,
    'epochs_check': 100,
    'save_every': 100,
    # 'hidden_dim_init': 8,
    # 'num_hidden_dims': 2,
    # 'hidden_dim_out': 50,
    # 'hidden_dim_other': 10,
    # 'num_hidden_dims2': 2,
    # 'hidden_dim_out2': 50,
    # 'hidden_dim_other2': 12,
    'learning_rate': 0.004,
    'times': 1,
}
dir_model = params_torque['base_dir_model']
with open(f'{dir_model}params.pkl', "rb") as tf:
    params_ = pickle.load(tf)
params_torque.update(params_)

params_data = {
    # 'ext': 'png',
    'path_data': f'{base_dir}_data_motor\\',
    'folder_image': 'geometry\\result\\image\\',
    # 'fnames': ['2D', '2U', 'V'],
    'fnames': ['2D', '2U', 'V', 'Nabla'],
    # 'image_size': 256,
    'data_number': 'dataset_number_scaled',
    'data_number2': 'dataset_number',
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

params_data['image_size'] = 224 if params_flux['modelname'] in vit_list else 256

#%%
class RegressionFlux(nn.Module):
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
        self.model_ft = prediction_model.model(params_flux['modelname'], params_flux['typename'], params_flux['pathmodel'])
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

class RegressionIronLoss(nn.Module):
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
            hidden_dim_out3=50,
            hidden_dim_other3=35,
            output_dim1=1,
            output_dim2=1,
            output_dim3=1,
            # weight_id=1.0,
            # weight_iq=1.0,
            # weight_speed=1.0,
            # learning_type='transfer_learning',
            activation_type='ReLU',
            **kwargs,
        ):
        super().__init__()
        ## init
        self.model_ft = prediction_model.model(params_ironloss['modelname'], params_ironloss['typename'], params_ironloss['pathmodel'])
        num_ftrs = 1000
        self.linear1 = nn.Linear(num_ftrs, hidden_dim_init)
        self.bn1 = nn.BatchNorm1d(hidden_dim_init)
        ## hysteresis
        hidden_dims = [hidden_dim_other]*num_hidden_dims
        hidden_dims[-1] = hidden_dim_out
        linear_list1 = []
        batch_norm_list1 = []
        hidden_dim_before = hidden_dim_init+current_dim+pm_material_dim+pm_temp_dim+speed_dim
        for hidden_dim in hidden_dims:
            linear_list1.append(nn.Linear(hidden_dim_before, hidden_dim))
            batch_norm_list1.append(nn.BatchNorm1d(hidden_dim))
            hidden_dim_before = hidden_dim
        self.linear_list1 = nn.ModuleList(linear_list1)
        self.batch_norm_list1 = nn.ModuleList(batch_norm_list1)
        self.out1 = nn.Linear(hidden_dim_before, output_dim1)
        ## joule
        hidden_dims2 = [hidden_dim_other2]*num_hidden_dims
        hidden_dims2[-1] = hidden_dim_out2
        linear_list2 = []
        batch_norm_list2 = []
        hidden_dim_before = hidden_dim_init+current_dim+pm_material_dim+pm_temp_dim+speed_dim
        for hidden_dim in hidden_dims2:
            linear_list2.append(nn.Linear(hidden_dim_before, hidden_dim))
            batch_norm_list2.append(nn.BatchNorm1d(hidden_dim))
            hidden_dim_before = hidden_dim
        self.linear_list2 = nn.ModuleList(linear_list2)
        self.batch_norm_list2 = nn.ModuleList(batch_norm_list2)
        self.out2 = nn.Linear(hidden_dim_before, output_dim2)
        ## pm_joule
        hidden_dims3 = [hidden_dim_other3]*num_hidden_dims
        hidden_dims3[-1] = hidden_dim_out3
        linear_list3 = []
        batch_norm_list3 = []
        hidden_dim_before = hidden_dim_init+current_dim+pm_material_dim+pm_temp_dim+speed_dim
        for hidden_dim in hidden_dims3:
            linear_list3.append(nn.Linear(hidden_dim_before, hidden_dim))
            batch_norm_list3.append(nn.BatchNorm1d(hidden_dim))
            hidden_dim_before = hidden_dim
        self.linear_list3 = nn.ModuleList(linear_list3)
        self.batch_norm_list3 = nn.ModuleList(batch_norm_list3)
        self.out3 = nn.Linear(hidden_dim_before, output_dim3)
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
        for i, (f, bn) in enumerate(zip(self.linear_list3, self.batch_norm_list3)):
            x3 = f(x3) if i > 0 else f(x)
            x3 = bn(x3)
            x3 = self.activation(x3)
        y1 = self.out1(x1)
        y2 = self.out2(x2)
        y3 = self.out3(x3)
        return y1, y2, y3 # hysteresis, joule, pm_joule

class RegressionTorque(nn.Module):
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
            output_dim1=1,
            activation_type='ReLU',
            **kwargs,
        ):
        super().__init__()
        ## init
        self.model_ft = prediction_model.model(params_torque['modelname'], params_torque['typename'], params_torque['pathmodel'])
        num_ftrs = 1000
        self.linear1 = nn.Linear(num_ftrs, hidden_dim_init)
        self.bn1 = nn.BatchNorm1d(hidden_dim_init)
        ## torque
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
        x = torch.cat([x,pm_material], axis=1)
        for i, (f, bn) in enumerate(zip(self.linear_list1, self.batch_norm_list1)):
            x1 = f(x1) if i > 0 else f(x)
            x1 = bn(x1)
            x1 = self.activation(x1)
        y1 = self.out1(x1)
        # y2 = self.out2(x2)
        return y1 #torque

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_flux = RegressionFlux(**params_flux).to(device)
model_flux.load_state_dict(torch.load(
    params_flux['base_dir_model']+params_flux['trained_model'], map_location=torch.device(device)
))
model_flux.eval()

model_ironloss = RegressionIronLoss(**params_ironloss).to(device)
model_ironloss.load_state_dict(torch.load(
    params_ironloss['base_dir_model']+params_ironloss['trained_model'], map_location=torch.device(device)
))
model_ironloss.eval()

model_torque = RegressionTorque(**params_torque).to(device)
model_torque.load_state_dict(torch.load(
    params_torque['base_dir_model']+params_torque['trained_model'], map_location=torch.device(device)
))
model_torque.eval()

#%%
class ImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        path_data = Path(params_data['path_data'])
        folder_image = params_data['folder_image']
        fnames = params_data['fnames']
        data_number = params_data['data_number']
        data_number2 = params_data['data_number2']
        data_image = params_data['data_image']
        # data_PM = params_data['data_PM']
        ## image
        image_size = params_data['image_size']
        self.paths = []
        for fname in fnames:
            df_image = pd.read_csv(path_data/f"{data_image}_{fname}.csv")
            path = [path_data/'raw'/fname/folder_image/p for p in df_image.values.flatten()]
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
            'speed': 'RPM',
            'flux': ['Psi_d','Psi_q'],
            'ironloss': ['W_e_pm', 'W_e_core', 'W_h_core'],
            'torque': 'T_avg',
            'pm_material': ['Coercivity','Remanence', 'Recoil', 'Drooping', 'Radius', 'PM_RESISTANCE'],
        }
        self.labels = {}
        for key in data_info.keys():
            self.labels[key] = pd.DataFrame()
        for fname in fnames:
            df = pd.read_csv(path_data/f'{data_number}_{fname}.csv')
            for key, val in data_info.items():
                self.labels[key] = pd.concat((self.labels[key], df[val]), axis=0)
        # key = 'pm_material'
        # self.labels[key] = pd.DataFrame()
        # for fname in fnames:
        #     self.labels[key] = pd.concat((
        #         self.labels[key], pd.read_csv(path_data/f'{data_PM}_{fname}.csv')
        #     ), axis=0)
        for key in self.labels.keys():
            self.labels[key] = self.labels[key].values
        data_info2 = {
            'flux': ['Psi_d','Psi_q'],
            'ironloss': ['W_e_pm', 'W_e_core', 'W_h_core'],
            'torque': 'T_avg',
        }
        self.labels2 = {}
        for key in data_info2.keys():
            self.labels2[key] = pd.DataFrame()
        for fname in fnames:
            df = pd.read_csv(path_data/f'{data_number2}_{fname}.csv')
            for key, val in data_info2.items():
                self.labels2[key] = pd.concat((self.labels2[key], df[val]), axis=0)        ## label_pm_material
        for key in self.labels2.keys():
            self.labels2[key] = self.labels2[key].values
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        current = self.labels['current'][index]
        speed = self.labels['speed'][index]
        pm_temp = self.labels['pm_temp'][index]
        x2 = np.concatenate([current, speed, pm_temp])
        # x2 = np.concatenate([current, pm_temp])
        pm_material = self.labels['pm_material'][index]
        psi_d, psi_q = self.labels['flux'][index]
        pm_joule, joule, hysteresis = self.labels['ironloss'][index]
        torque = self.labels['torque'][index]

        psi_d2, psi_q2 = self.labels2['flux'][index]
        pm_joule2, joule2, hysteresis2 = self.labels2['ironloss'][index]
        torque2 = self.labels2['torque'][index]

        return (
            self.transform(img),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(pm_material, dtype=torch.float32),
            torch.tensor(psi_d, dtype=torch.float32),
            torch.tensor(psi_q, dtype=torch.float32),
            torch.tensor(hysteresis, dtype=torch.float32),
            torch.tensor(joule, dtype=torch.float32),
            torch.tensor(pm_joule, dtype=torch.float32),
            torch.tensor(torque, dtype=torch.float32),
            psi_d2,
            psi_q2,
            hysteresis2,
            joule2,
            pm_joule2,
            torque2,
        )

# def set_data_src():
#     dataset = ImageDataset()
#     n_samples = len(dataset)
#     train_size = int(n_samples*0.8)
#     valid_size = n_samples - train_size
#     train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
#     train_dataloader = DataLoader(
#         train_dataset,
#         )
#     valid_dataloader = DataLoader(
#         valid_dataset,
#         )
#     return train_dataloader, valid_dataloader
# train_loader, valid_loader = set_data_src()

dataset = ImageDataset()
n_samples = len(dataset)
train_size = int(n_samples*0.8)
valid_size = n_samples - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

#%%
preds_psi_dq_all = []
data_psi_dq_all = []
preds_ironloss_all = []
data_ironloss_all = []
preds_torque_all = []
data_torque_all = []

data_psi_dq_all2 = []
data_ironloss_all2 = []
data_torque_all2 = []

data_idq_all = []

# indices = range(0, len(dataset), 1000)
indices = range(0, len(valid_dataset), 100)
for i in tqdm(indices):
# for data in tqdm(valid_dataset):
    # data = dataset[i]
    data = valid_dataset[i]
    img = data[0].to(device).unsqueeze(0)
    x2 = data[1].to(device).unsqueeze(0)
    x2_f = torch.cat((x2[:,:2], x2[:,3:]),axis=1)
    pm_material = data[2].to(device).unsqueeze(0)
    
    data_psi_dq_all.append(
        torch.tensor(
            data[3:5]
        ).to('cpu').detach().numpy().copy() 
    )
    data_ironloss_all.append(
        torch.tensor(
            data[5:8]
        ).to('cpu').detach().numpy().copy() 
    )
    data_torque_all.append(
        torch.tensor(
            data[8]
        ).to('cpu').detach().numpy().copy() 
    )
    preds_psi_dq_all.append(
        torch.tensor(
            model_flux(img, x2_f, pm_material)
        ).to('cpu').detach().numpy().copy() 
    )
    preds_ironloss_all.append(
        torch.tensor(
            model_ironloss(img, x2, pm_material)
        ).to('cpu').detach().numpy().copy() 
    )
    preds_torque_all.append(
        torch.tensor(
            model_torque(img, x2_f, pm_material)
        ).to('cpu').detach().numpy().copy()[0]
    )
    data_psi_dq_all2.append(
        data[9:11]
    )
    data_ironloss_all2.append(
        data[11:14]
    )
    data_torque_all2.append(
        data[14]
    )

    data_idq_all.append(
        torch.tensor(
            x2[0,:2]
        ).to('cpu').detach().numpy().copy() 
    )
preds_psi_dq_all = np.array(preds_psi_dq_all)
data_psi_dq_all = np.array(data_psi_dq_all)
preds_ironloss_all = np.array(preds_ironloss_all)
data_ironloss_all = np.array(data_ironloss_all)
preds_torque_all = np.array(preds_torque_all)
data_torque_all = np.array(data_torque_all)

data_psi_dq_all2 = np.array(data_psi_dq_all2)
data_ironloss_all2 = np.array(data_ironloss_all2)
data_torque_all2 = np.array(data_torque_all2)

data_idq_all = np.array(data_idq_all)

# %%
preds_all = np.hstack((preds_psi_dq_all,preds_ironloss_all,preds_torque_all)).T
data_all = np.hstack((data_psi_dq_all,data_ironloss_all,data_torque_all)).T
# p = preds_psi_dq_all[:,0]
# d = data_psi_dq_all[:,0]

for p, d in zip(preds_all, data_all):
    plt.plot(p, d, 'bo', ms=3)
    plt.plot([d.min(), d.max()], [d.min(), d.max()], 'k--')
    r2 = r2_score(d, p)
    mse = mean_squared_error(d, p)
    plt.title(f'r2: {round(r2, 2)}, mse: {round(mse,3)}')
    plt.show()

#%%
df_sp = pd.read_csv('..\\_data_motor\\dataset_scaling_parameter_all.csv')
df_sp.index = ['mean','std']

data_all2 = np.hstack((data_psi_dq_all2,data_ironloss_all2,data_torque_all2)).T

def _scaling(x, col):
    return (np.array(x)-df_sp.loc['mean',col])/df_sp.loc['std',col]
def _unscaling(x, col):
    return np.array(x)*df_sp.loc['std',col]+df_sp.loc['mean',col]
names = ['Psi_d','Psi_q','W_h_core','W_e_core','W_e_pm','T_avg']

for p, d, n in zip(data_all, data_all2, names):
    d_ = _scaling(d, n)
    plt.plot(p, d_, 'bo', ms=3)
    plt.plot([d_.min(), d_.max()], [d_.min(), d_.max()], 'k--')
    r2 = r2_score(d_, p)
    mse = mean_squared_error(d_, p)
    plt.title(f'r2: {round(r2, 2)}, mse: {round(mse,3)}')
    plt.show()


#%%
for p, d, n in zip(preds_all, data_all, names):
    p_ = _unscaling(p, n)
    d_ = _unscaling(d, n)
    plt.plot(p_, d_, 'bo', ms=3)
    plt.plot([d_.min(), d_.max()], [d_.min(), d_.max()], 'k--')
    r2 = r2_score(d_, p_)
    mse = mean_squared_error(d_, p_)
    plt.title(f'r2: {round(r2, 2)}, mse: {round(mse,3)}')
    plt.show()

#%%
Pn = 4
id_ = data_idq_all[:,0]
iq_ = data_idq_all[:,1]
id__ = _unscaling(id_, 'id')
iq__ = _unscaling(iq_, 'iq')
psi_d = preds_all[0]
psi_q = preds_all[1]
psi_d_ = _unscaling(psi_d, 'Psi_d')
psi_q_ = _unscaling(psi_q, 'Psi_q')
torque = Pn*(iq__*psi_d_-id__*psi_q_)

# d_ = _unscaling(d, n)
d = data_all2[5]
plt.plot(torque, d, 'bo', ms=3)
plt.plot([d.min(), d.max()], [d.min(), d.max()], 'k--')
r2 = r2_score(d, torque)
mse = mean_squared_error(d, torque)
plt.title(f'r2: {round(r2, 2)}, mse: {round(mse,3)}')
plt.show()


#%%
print(data_all)
#%%

#%%

#%%
