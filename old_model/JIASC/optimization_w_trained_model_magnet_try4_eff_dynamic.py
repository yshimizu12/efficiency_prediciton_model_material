#%%
# import json
# import csv
import math
# import multiprocessing
# import datetime
from pathlib import Path
from tqdm import tqdm
import time
import pickle
import re
import datetime

import cv2
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
from lightweight_gan.lightweight_gan import Trainer

import evaluation
import prediction_model

#%%
base_dir = '..\\'

params_flux = {
    'modelname':'swin_t',
    'typename':'transfer_learning',
    'pathmodel':None,
    'base_dir_model':f'{base_dir}_result\\material\\2_brhc2\\2305162137_pred_flux_brhc_swin_t_w_params\\',
    # 'base_dir_model':f'{base_dir}_result\\material\\1_brhc\\2305112018_pred_flux_brhc_swin_t_w_params_torque_MSE3\\',
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
    'base_dir_model':f'{base_dir}_result\\material\\2_brhc2\\2305162107_pred_ironloss_brhc_swin_t_w_params\\',
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

params_gan = {
    'path_generator': f'{base_dir}_GAN\\230518_data_aug\\',
    'name': '2D-V-Nabla-2U',
    'load_from': 15,
}

params_data = {
    # 'ext': 'png',
    'path_data': f'{base_dir}_data_motor',
    'folder_image': 'geometry\\result\\image\\',
    # 'fnames': ['2D', '2U', 'V'],
    'fnames': ['2D', '2U', 'V', 'Nabla'],
    # 'image_size': 256,
    'data_number': 'dataset_number_scaled',
    'data_class': 'dataset_class',
    'data_image': 'dataset_image',
    'data_PM': 'dataset_material_PM_dummies',
    'scaling_parameters': 'dataset_scaling_parameter',
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

    def forward(self, image, parameters, pm_material):
        x = self.forward1(image)
        y1, y2 = self.forward2(x, parameters, pm_material)
        return y1, y2

    def forward1(self, image):
        ## image
        x = self.model_ft(image)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation(x)
        return x

    def forward2(self, x, parameters, pm_material):
        ## image
        x = torch.cat([x,parameters], axis=1)
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
        return y1, y2

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
        x = self.forward1(image)
        y1, y2, y3 = self.forward2(x, parameters, pm_material)
        return y1, y2, y3 # hysteresis, joule, pm_joule

    def forward1(self, image):
        x = self.model_ft(image)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation(x)
        return x

    def forward2(self, x, parameters, pm_material):
        x = torch.cat([x,parameters], axis=1)
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
        return y1, y2, y3

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

#%%
# trainer = Trainer(
#     disc_output_size = 1,
#     image_size = 256,
#     use_aim=False
# )
model = Trainer(
    models_dir = params_gan['path_generator'] + 'models',
    results_dir = params_gan['path_generator'] + 'results',
    name = params_gan['name'],
    disc_output_size = 1,
    image_size = 256,
    use_aim=False
)
model.load(params_gan['load_from'])
GAN = model.GAN.eval()
# trainer.init_GAN()
# GAN = trainer.GAN
# GAN.load_state_dict(torch.load(params_gan['path_generator'])['GAN'])

#%%
x = np.ones(256)
x = np.random.randn(256)
generated_image_ = GAN.G(torch.from_numpy(x.reshape(1,-1)).to(device=device,dtype=torch.float)).clamp_(0., 1.)
generated_image_ = generated_image_.cpu().detach().numpy()[0].transpose(1,2,0)

plt.imshow(generated_image_)


# %%
def clear_blurred_image(img):
    img_array = np.array(img)
    max_vals = np.max(img_array, axis=2)
    red_pixels = np.where(img_array[:, :, 0] == max_vals)
    green_pixels = np.where(img_array[:, :, 1] == max_vals)
    blue_pixels = np.where(img_array[:, :, 2] == max_vals)
    img_array[red_pixels[0], red_pixels[1], :] = [255, 0, 0]
    img_array[green_pixels[0], green_pixels[1], :] = [0, 255, 0]
    img_array[blue_pixels[0], blue_pixels[1], :] = [0, 0, 255]
    return img_array

def reconstruct_motor_image(img_polar, n_area=300, n_out=693, n_in=250):
    m = img_polar.shape[0]
    img_reconst = cv2.rotate(img_polar, cv2.ROTATE_90_CLOCKWISE)
    img_reconst = cv2.resize(img_reconst, dsize=(m-int(n_area/n_out*m)-1,int(m/2)),interpolation=cv2.INTER_NEAREST)
    img_red = np.zeros((int(m/2), m-int(n_in/n_out*m)-1, 3), np.uint8)
    img_red[:, :] = (255, 0, 0)
    x_offset = int(n_area/n_out*m)-int(n_in/n_out*m)
    img_red[:, x_offset:] = img_reconst
    img_reconst = img_red.copy()
    img_reconst2 = cv2.flip(img_reconst, 0)
    img_reconst = np.vstack((img_reconst, img_reconst2))
    img_reconst = np.vstack((img_reconst, img_reconst))
    img_reconst = np.vstack((img_reconst, img_reconst2))
    img_black = np.zeros((8*m, m, 3), np.uint8)
    x_offset = m - img_reconst.shape[1]
    y_offset = 0
    img_black[y_offset:y_offset+img_reconst.shape[0], x_offset:x_offset+img_reconst.shape[1]] = img_reconst
    img_reconst = img_black.copy()
    flags = cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
    img_reconst = cv2.warpPolar(img_reconst, (n_out, n_out), (0, 0), n_out, flags)
    img_reconst = cv2.flip(img_reconst, 0)
    # black_pixels = np.where((img_reconst == [0, 0, 0]).all(axis=2))
    # img_reconst[black_pixels] = [255, 255, 255]
    # img_reconst[img_reconst<0] = 0
    # img_reconst[img_reconst>200] = 255
    # img_reconst[np.where((img_reconst == [255, 0, 0]).all(axis=2))] = [200, 200, 200]
    # img_reconst[np.where((img_reconst == [0, 0, 255]).all(axis=2))] = [0, 0, 0]
    # img_reconst[np.where((img_reconst == [0, 255, 0]).all(axis=2))] = [255, 255, 255]
    black_pixels = np.where((img_reconst == [0, 0, 0]).all(axis=2))
    max_vals = np.max(img_reconst, axis=2)
    red_pixels = np.where(img_reconst[:, :, 0] == max_vals)
    green_pixels = np.where(img_reconst[:, :, 1] == max_vals)
    blue_pixels = np.where(img_reconst[:, :, 2] == max_vals)
    img_reconst[red_pixels[0], red_pixels[1], :] = [200, 200, 200]
    img_reconst[green_pixels[0], green_pixels[1], :] = [255, 255, 255]
    img_reconst[blue_pixels[0], blue_pixels[1], :] = [0, 0, 0]
    img_reconst[black_pixels] = [255, 255, 255]
    return img_reconst


#%%
# params_prediction = {
#     'Ie_max': 300, # 0~800/(2**0.5)
#     'RPM_max': 14000, # 0~30000
#     # 'TEMP_PM': 40, # 0~200
#     # 'PM_material': 'NMX-S49CH',
#     'Vdc': 650,
#     'Ra': 0.1,
#     'Pn': 4,
#     'device': device,
#     'include_pm_joule': False,
#     'path_param_scaling': params_data['path_data']+'\\'+params_data['scaling_parameters']+'_all.csv',
# }
# df_sp = pd.read_csv(params_prediction['path_param_scaling'])
# df_sp.index = ['mean','std']
# params_prediction['param_scaling'] = df_sp

# def _scaling(x, col):
#     return (np.array(x)-df_sp.loc['mean',col])/df_sp.loc['std',col]
# def _unscaling(x, col):
#     return np.array(x)*df_sp.loc['std',col]+df_sp.loc['mean',col]

# data_dir = Path(f"{params_data['path_data']}\\raw\\_common_setting\\b-h_PM")
# pattern = r"b-h_(.+)\.csv"
# PM_names = [re.search(pattern, p.name).group(1) for p in data_dir.glob('*.csv')]
# PM_data = {
#     n: pd.read_csv(p) for n, p in zip(PM_names, data_dir.glob('*.csv'))
# }
# PM_class = {
#     'NMX':[],
#     'R':[],
# }
# for n in PM_names:
#     if n.startswith('NMX'):
#         PM_class['NMX'].append(n)
#     elif n.startswith('R'):
#         PM_class['R'].append(n)

# params_prediction['PM_data'] = PM_data
# params_prediction['PM_class'] = PM_class

# evaln = evaluation.Evaluate(
#     model_flux=model_flux,
#     model_ironloss=model_ironloss,
#     **params_prediction)

# #%%
# path_img = 'D:\\program\\github\\_data_motor\\raw\\2D\\geometry\\result\\image\\000000.png'
# img = Image.open(path_img)
# img = np.array(img)
# plt.imshow(img)
# plt.axis('off')
# plt.show()
# # img_recon = clear_blurred_image(img)
# # plt.imshow(img_recon)
# # plt.axis('off')
# # plt.show()
# img_recon = reconstruct_motor_image(img)
# plt.imshow(img_recon)
# plt.axis('off')
# plt.show()

# #%%

# rotor_image_tensor = torch.from_numpy(np.array([
#     img.transpose(2,0,1).astype(np.float32)
# ])).clone().to(device)

# df_data = pd.read_csv("D:\\program\\github\\_data_motor\\dataset_number_2D.csv")
# df_data_scaled = pd.read_csv("D:\\program\\github\\_data_motor\\dataset_number_scaled_2D.csv")
# df_data_pm = pd.read_csv("D:\\program\\github\\_data_motor\\dataset_class_2D.csv")

# i = 0
# Ia,beta = df_data.loc[i][['Amp','Beta']]
# Ia = Ia*(3/2)**0.5

# material_PM = df_data_pm.loc[i]['material_PM']
# TEMP_PM = df_data.loc[i]['PM_TEMP']
# evaln._set_PM_material_parameter(TEMP_PM, material_PM)

# encoded_img = evaln._calc_encoded_img(rotor_image_tensor)
# torque_pred = evaln._torque_calculation(Ia, beta, encoded_img[0])
# print(torque_pred, df_data.loc[0]['T_avg'])

# psi_dq_pred = evaln._flux_calculation(Ia, beta, encoded_img[0])
# print(np.array(psi_dq_pred).flatten(), df_data.loc[i][['Psi_d','Psi_q']].values)

# #%%
# psi_dq_pred = []
# psi_dq_data = []
# for i in range(30):
#     Ia,beta = df_data.loc[i][['Amp','Beta']]
#     Ia = Ia*(3/2)**0.5

#     material_PM = df_data_pm.loc[i]['material_PM']
#     TEMP_PM = df_data.loc[i]['PM_TEMP']
#     evaln._set_PM_material_parameter(TEMP_PM, material_PM)

#     encoded_img = evaln._calc_encoded_img(rotor_image_tensor)
#     psi_dq_pred.append(np.array(evaln._flux_calculation(Ia, beta, encoded_img[0])).flatten())
#     psi_dq_data.append(df_data.loc[i][['Psi_d','Psi_q']].values)

# psi_dq_pred = np.array(psi_dq_pred)
# psi_dq_data = np.array(psi_dq_data)

# for j in range(2):
#     d = psi_dq_data[:,j]
#     p = psi_dq_pred[:,j]
#     plt.plot(p, d, 'bo', ms=3)
#     plt.plot([d.min(), d.max()], [d.min(), d.max()], 'k--')
#     r2 = r2_score(d, p)
#     mse = mean_squared_error(d, p)
#     plt.title(f'r2: {round(r2, 2)}, mse: {round(mse,3)}')
#     plt.show()

# #%%

# beta = np.arange(0,91,1)
# for i in range(3):
#     material_PM = df_data_pm.loc[i]['material_PM']
#     TEMP_PM = df_data.loc[i]['PM_TEMP']
#     evaln._set_PM_material_parameter(TEMP_PM, material_PM)
#     torque = [evaln._torque_calculation(300, b, encoded_img[0]) for b in beta]
#     plt.plot(beta, torque)
#     plt.show()

# #%%
# params = {
#     'Ie_max': 134, # 0~800/(2**0.5)
#     'RPM_max': 14000, # 0~30000
#     'Vdc': 650,
#     'TEMP_PM': 20, # 0~200
#     'PM_material': 'NMX-34EH',
#     'include_pm_joule': False,
# }
# evaln._elec_params_setting(params['Ie_max'], params['Vdc'])
# evaln.Nmax = params['RPM_max']
# evaln._set_PM_material_parameter(params['TEMP_PM'], params['PM_material'])

# x = np.ones(256)
# x = np.random.randn(256)
# generated_image_ = GAN.G(torch.from_numpy(x.reshape(1,-1)).to(device=device,dtype=torch.float)).clamp_(0., 1.)
# img = generated_image_.cpu().detach().numpy()[0].transpose(1,2,0)
# path_img = 'D:\\program\\github\\_data_motor\\raw\\2D\\geometry\\result\\image\\000000.png'
# img = Image.open(path_img)
# img = np.array(img)

# plt.imshow(img)
# plt.show()

# start = time.time()
# evaln.evaluation(img, 'hoge')
# print(time.time() - start)
# evaln.create_efficiency_map(rotor_image_tensor)
#%%
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 13

plt.rcParams["mathtext.fontset"] = 'stix'

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.grid'] = True

plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.framealpha'] = 1
plt.rcParams['legend.edgecolor'] = 'black'

plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.right'] = False

# import datetime

# dt_now = datetime.datetime.now()
# date = dt_now.strftime('%Y%m%d')[2:]

#%%
import warnings
warnings.resetwarnings()
warnings.simplefilter('ignore', DeprecationWarning)

#%%
def calc_pareto_front(f):
    f_sorted_by_f1 = f[np.argsort(f[:,0])]    
    pareto_front = [f_sorted_by_f1[0]]
    for pair in f_sorted_by_f1[1:]:
        if pair[1] <= pareto_front[-1][1]:
            pareto_front = np.vstack((pareto_front,pair))
    return pareto_front    

#%%
import cv2

def judge_topology(img):
    img_h = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,0]
    thre, region_h_min = cv2.threshold(img_h, 200, 1, cv2.THRESH_BINARY_INV)
    thre, region_h_max = cv2.threshold(img_h, 300, 1, cv2.THRESH_BINARY)
    region_pm = (region_h_min==region_h_max).astype(np.uint8)
    contours_pm, _ = cv2.findContours(region_pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pm = list(filter(lambda x: cv2.contourArea(x) > 100, contours_pm))
    if len(contours_pm) == 1: return 1
    elif len(contours_pm) == 2: return 2
    elif len(contours_pm) == 3: return 3  

#%%
class Optimize:
    def __init__(self, params_prediction, params_optimization):
        self.params_prediction = params_prediction
        self.params_optimization = params_optimization
    # def optimize(self, Ie_max, Vdc, torque1, torque2, efficiency1, efficiency2):
    #     self.Ie_max = Ie_max
    #     self.Vdc = Vdc
    #     self.torque1 = torque1
    #     self.torque2 = torque2
    #     self.efficiency1 = efficiency1
    #     self.efficiency2 = efficiency2
    #     self.params_prediction['Ie_max'] = float(Ie_max)
    #     self.params_prediction['Vdc'] = float(Vdc)
    #     required_torque_points = np.array([
    #         [float(x.strip()) for x in torque1.split(',')], 
    #         [float(x.strip()) for x in torque2.split(',')],
    #     ])
    #     evaluate_efficiency_points = np.array([
    #         [float(x.strip()) for x in efficiency1.split(',')], 
    #         [float(x.strip()) for x in efficiency2.split(',')],
    #     ])
    #     self.params_optimization['required_torque_points'] = required_torque_points
    #     self.params_optimization['evaluate_efficiency_points'] = evaluate_efficiency_points
    #     self.params_optimization['n_obj'] = evaluate_efficiency_points.shape[1]
    #     self.params_optimization['n_constr'] = required_torque_points.shape[1]

    #     self.opt = evaluation.Optimize(
    #         params_prediction=self.params_prediction,
    #         params_optimization=self.params_optimization,
    #     )
    #     self.opt.optimize(seed=params_optimization['seed']) #0,3
    #     # opt.show_best_result()
    #     # self.opt = opt
    def optimize(self, Ie_max, Vdc, TEMP_PM, PM_material, torques, efficiencies):
        self.params_prediction['Ie_max'] = float(Ie_max)
        self.params_prediction['Vdc'] = float(Vdc)
        self.params_prediction['TEMP_PM'] = float(TEMP_PM)
        self.params_prediction['PM_material'] = PM_material
        required_torque_points = np.array(torques)
        evaluate_efficiency_points = np.array(efficiencies)
        self.params_optimization['required_torque_points'] = required_torque_points
        self.params_optimization['evaluate_efficiency_points'] = evaluate_efficiency_points
        self.params_optimization['n_obj'] = evaluate_efficiency_points.shape[1]
        self.params_optimization['n_constr'] = required_torque_points.shape[1]

        self.opt = evaluation.Optimize(
            params_prediction=self.params_prediction,
            params_optimization=self.params_optimization,
        )
        self.opt.optimize(seed=params_optimization['seed']) #0,3
        # opt.show_best_result()
        # self.opt = opt

#%%
params_prediction = {
    'Ie_max': 200, # 0~800/(2**0.5)
    'RPM_max': 14000, # 0~30000
    'TEMP_PM': 40, # 0~200
    'PM_material': 'NMX-34EH',
    'Vdc': 650,
    'Ra': 0.1,
    'Pn': 4,
    'device': device,
    'include_pm_joule': False,
    'path_param_scaling': params_data['path_data']+'\\'+params_data['scaling_parameters']+'_all.csv',
}
df_sp = pd.read_csv(params_prediction['path_param_scaling'])
df_sp.index = ['mean','std']
params_prediction['param_scaling'] = df_sp
data_dir = Path(f"{params_data['path_data']}\\raw\\_common_setting\\b-h_PM")
pattern = r"b-h_(.+)\.csv"
PM_names = [re.search(pattern, p.name).group(1) for p in data_dir.glob('*.csv')]
PM_data = {n: pd.read_csv(p) for n, p in zip(PM_names, data_dir.glob('*.csv'))}
PM_class = {'NMX':[], 'R':[],}
for n in PM_names:
    if n.startswith('NMX'): PM_class['NMX'].append(n)
    elif n.startswith('R'): PM_class['R'].append(n)
params_prediction['PM_data'] = PM_data
params_prediction['PM_class'] = PM_class

n_var = GAN.latent_dim
params_optimization = {
    'model_flux': model_flux,
    'model_ironloss': model_ironloss,
    'GAN': GAN,
    'n_var': n_var,
    'xl': np.ones(n_var)*-1000,
    'xu': np.ones(n_var)*1000,
}
params_optimization['pop_size'] = 100
params_optimization['n_offsprings'] = 10
params_optimization['n_termination'] = 50
params_optimization['verbose'] = True
params_optimization['seed'] = 3

f = Optimize(params_prediction=params_prediction, params_optimization=params_optimization)

#%%
# f.optimize(
#     Ie_max='250', Vdc='650', 
#     torque1='280, 3000', torque2='55, 11000', 
#     efficiency1='100, 3500', efficiency2='50, 11000'
# )
# f.optimize(
#     Ie_max=250, Vdc=650, TEMP_PM=40, PM_material='NMX-34EH',
#     torques=[[280, 3000],[55, 11000]], 
#     efficiencies=[[100, 3500],[50, 11000]]
# )


#%%
def find_min_m(n):
    m = math.ceil(math.sqrt(n))
    while m**2 < n: m += 1
    return m


#%%
PM_material_list = [
    'NMX-K30ER',
    'NMX-34EH',
    'NMX-39EH',
    'NMX-43SH',
    'NMX-S49CH',
    'NMX-S52',
    'R26HE',
    'R30',
    'R32HS',
    'R33H',
]
PM_TEMP_list = [20,40,60,80,100,120,140,160,180]
# PM_TEMP_list = [180,]

save_dir = Path('optimization_results') / 'change_PM'
dt = datetime.datetime.now().strftime('%Y%m%d%H%M%S')[2:]

params_optimization['pop_size'] = 100
params_optimization['n_offsprings'] = 10
params_optimization['n_termination'] = 50

for PM_mat in PM_material_list:
    for PM_temp in PM_TEMP_list:
        f = Optimize(params_prediction=params_prediction, params_optimization=params_optimization)

        f.optimize(
            Ie_max=300, Vdc=650, TEMP_PM=PM_temp, PM_material=PM_mat,
            torques=[[280, 3000],[55, 11000]], 
            efficiencies=[[50, 3500],[50, 11000]],
        )

        f_1 = f.opt.result["F"]
        x_1 = f.opt.result["X"]
        g_1 = f.opt.result["G"]
        hv_1 = f.opt.result["HV"]
        cond_1 = f.opt.result["COND"]
        loss_1 = f.opt.result["LOSS"]


        (save_dir / f'{dt}\\mat_{PM_mat}_temp_{PM_temp}').mkdir(parents=True, exist_ok=True)
        np.save(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_fitness_2eff_coef.npy', np.array(f_1))
        np.save(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_xgeom_2eff_coef.npy', np.array(x_1))
        np.save(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_constraint_2eff_coef.npy', np.array(g_1))
        np.save(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_hv_2eff_coef.npy', np.array(hv_1))
        np.save(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_conditions_2eff_coef.npy', np.array(cond_1))
        np.save(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_losses_2eff_coef.npy', np.array(loss_1))

        params_save = params_optimization.copy()
        del_list = [
            'model_flux', 'model_ironloss', 'GAN', 'xl', 'xu',]
        for d in del_list:
            params_save.pop(d)
        with open(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\optimization_params.pkl', "wb") as tf:
            pickle.dump(params_save,tf)

#%%
PM_material_list = [
    'NMX-K30ER',
    'NMX-34EH',
    'NMX-39EH',
    'NMX-43SH',
    'NMX-S49CH',
    'NMX-S52',
    'R26HE',
    'R30',
    'R32HS',
    'R33H',
]
# PM_TEMP_list = [20,40,60,80,100,120,140,160,180]
PM_TEMP_list = [60,]
# PM_TEMP_list = [180,]
# Ie_list = [210]
save_dir = Path('optimization_results') / 'change_PM'
dt = '230526224322'
# x_1_all = []
pf_tmp_all = {}

for PM_mat in PM_material_list:
    for PM_temp in PM_TEMP_list:
        f_1 = np.load(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_fitness_2eff_coef.npy')
        x_1 = np.load(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_xgeom_2eff_coef.npy')
        g_1 = np.load(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_constraint_2eff_coef.npy')
        hv_1 = np.load(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_hv_2eff_coef.npy')
        cond_1 = np.load(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_conditions_2eff_coef.npy')
        loss_1 = np.load(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_losses_2eff_coef.npy')

        check_constraint_last = ~(g_1[-1]>0).sum(axis=1).astype(bool)
        # x_1_all.append(x_1[-1][check_constraint_last])

        PA = f.params_optimization['required_torque_points'][0]
        PB = f.params_optimization['required_torque_points'][1]
        # fig, ax = plt.subplots(figsize=(8/2.54,5/2.54))
        fig, ax = plt.subplots(figsize=(13/2.54,8/2.54))

        g_1_ = g_1[-1][check_constraint_last]
        alpha = 1.05
        # ax.plot(pf_tmp[:,0]*14.63, pf_tmp[:,1]*-197, c='c', linewidth=1)
        ax.plot([140,330],[PB[0]*alpha,PB[0]*alpha],'b--', linewidth=1)
        ax.plot([PA[0]*alpha,PA[0]*alpha],[25,70],'b--', linewidth=1)
        # ax.plot([140,330],[PB[0],PB[0]],'b--', linewidth=1)
        # ax.plot([PA[0],PA[0]],[25,70],'b--', linewidth=1)
        ax.plot((-np.array(g_1)[:,:,0]*PA[0]+PA[0]*alpha).flatten(),
                (-np.array(g_1)[:,:,1]*PA[0]+PB[0]*alpha).flatten(),
                'o', c='k', ms=1, label='All pop.')
        ax.plot(-np.array(g_1)[-1,:,0]*PA[0]+PA[0]*alpha,
                -np.array(g_1)[-1,:,1]*PA[0]+PB[0]*alpha,
                's', c='g', ms=2, label='Last pop.')
        ax.plot(-np.array(g_1_)[:,0]*PA[0]+PA[0]*alpha,
                -np.array(g_1_)[:,1]*PA[0]+PB[0]*alpha,
                '^', c='r', ms=2, label='Last pop.')

        ax.set_xlabel('Torque at P_A (Nm)')
        ax.set_ylabel('Torque at P_B (Nm)')

        ax.text(140,40*1.07 ,'Constraint P_B',color='b', fontsize=7)
        ax.text(197*1.05,26 ,'Constraint P_A',color='b', rotation=-90, fontsize=7)
        ax.legend(loc='upper left',fontsize=7)

        fig.tight_layout()

    # for Ie in Ie_list:
        # g_1 = np.load(f'{save_dir}\\Ie_{Ie}\\{dt}_2DVnabla_constraint_2eff_coef.npy')
        # check_constraint_last = ~(g_1[-1]>0).sum(axis=1).astype(bool)
        # loss_1 = np.load(f'{save_dir}\\Ie_{Ie}\\{dt}_2DVnabla_losses_2eff_coef.npy')
        s = np.array(loss_1).shape

        fig, ax = plt.subplots(figsize=(12/2.54,8/2.54))
        ax.plot(np.array(loss_1)[:,:,:,:2].sum(axis=3).reshape(s[0]*s[1], s[2])[:,0],
                np.array(loss_1)[:,:,:,:2].sum(axis=3).reshape(s[0]*s[1], s[2])[:,1],
                'o', c='k', ms=1, label='All population')
        ax.plot(np.array(loss_1[-1])[:,:,:2].sum(axis=2).reshape(s[1], s[2])[:,0],
                np.array(loss_1[-1])[:,:,:2].sum(axis=2).reshape(s[1], s[2])[:,1],
                's', c='g', ms=3, label='Last population')
        loss_1_ = loss_1[-1][:,:,:2].sum(axis=2)
        loss_1_ok = loss_1_[check_constraint_last]
        err = False
        try:
            if check_constraint_last.sum()>4:
                pf_tmp = np.array(calc_pareto_front(loss_1_ok))
            else:
                pf_tmp = loss_1_ok
            pf_tmp_all[f'{PM_mat}_{PM_temp}'] = pf_tmp
            ax.plot(pf_tmp[:,0], pf_tmp[:,1], c='b', linewidth=1)
        except:
            err = True
        if pf_tmp.shape[0]==0: err = True
        ax.plot(np.array(loss_1_ok)[:,0],
                np.array(loss_1_ok)[:,1],
                '^', c='r', ms=3, label='Last population')
        ax.set_xlabel(f'loss at P_1 (W)')
        ax.set_ylabel(f'loss at P_2 (W)')

        ax.legend(loc='best',fontsize=13)
        # fig.tight_layout()
        plt.show()

        if err:
            print('no effective solution\n\n')
        else:
            pf_tmp_images = []
            pf_tmp_conditions = []
            n = -1
            res = find_min_m(pf_tmp.shape[0])
            fig, axes = plt.subplots(res, res, figsize=(10, 10))
            for i in range(pf_tmp.shape[0]):
                latent_variable = x_1[n][
                    np.where(
                        # np.array(loss_1_)==pf_tmp[pf_tmp[:,1]<-0.5][i][0]
                        np.array(loss_1_)==pf_tmp[i][0]
                    )[0][0]]
                pf_tmp_conditions.append(cond_1[n][
                    np.where(
                        np.array(loss_1_)==pf_tmp[i][0]
                    )[0][0]]
                )
            #     im = generate_truncated(GAN.G, torch.from_numpy(latent_variable.reshape(1,-1)).to(device=device, dtype=torch.float) )
                im = GAN.G(torch.from_numpy(latent_variable.reshape(1,-1)).to(device=device, dtype=torch.float) )
                im_np = im.to('cpu').detach().numpy()[0].transpose(1,2,0)
                im_np[im_np>1.]=1.
                im_np[im_np<0.]=0.
                pf_tmp_images.append(im_np)
                # plt.imshow( im_np )
                # plt.show()

                img_recon = clear_blurred_image(im_np)
                img_recon = reconstruct_motor_image(img_recon)
                if pf_tmp.shape[0]==1:
                    axes.imshow(img_recon)
                    axes.axis('off')
                else:
                    axes.flatten()[i].imshow(img_recon)
            if pf_tmp.shape[0]!=1:
                for i in range(res**2):
                    axes.flatten()[i].axis('off')
                    # plt.imshow(img_recon)
                    # plt.axis('off')
                    # plt.show()
            plt.show()

for key in pf_tmp_all.keys():
    if pf_tmp.shape[0]==1:
        plt.plot(pf_tmp_all[key][:,0], pf_tmp_all[key][:,1], 'o', label=key)
    else:
        plt.plot(pf_tmp_all[key][:,0], pf_tmp_all[key][:,1], linewidth=1, label=key)
plt.legend(loc='best', fontsize=7)
plt.show()

#%%
PM_material_list = [
    'NMX-K30ER',
    'NMX-34EH',
    'NMX-39EH',
    'NMX-43SH',
    'NMX-S49CH',
    'NMX-S52',
    'R26HE',
    'R30',
    'R32HS',
    'R33H',
]
PM_TEMP_list = [20,40,60,80,100,120,140,160,180]

# for key in pf_tmp_all.keys():
for PM_temp in PM_TEMP_list:
    for PM_mat in PM_material_list:
        key = f'{PM_mat}_{PM_temp}'
        if pf_tmp.shape[0]==1:
            plt.plot(pf_tmp_all[key][:,0], pf_tmp_all[key][:,1], 'o', label=key)
        else:
            plt.plot(pf_tmp_all[key][:,0], pf_tmp_all[key][:,1], linewidth=1, label=key)
    # plt.title(PM_mat)
    plt.title(PM_temp)
    plt.legend(loc='best', fontsize=7)
    plt.xlim([1200, 1700])
    plt.ylim([3000, 3700])
    plt.show()

#%%
PM_material_list = [
    'NMX-39EH',
    'NMX-S52',
    'R33H',
]
PM_TEMP_list = [20,40,60,80,100,120,140,160,180]
# PM_TEMP_list = [60,]
# PM_TEMP_list = [180,]
# Ie_list = [210]
save_dir = Path('optimization_results') / 'change_PM'
dt = '230526224322'
# x_1_all = []

for PM_mat in PM_material_list:
    pf_tmp_all = {}
    for PM_temp in PM_TEMP_list:
        f_1 = np.load(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_fitness_2eff_coef.npy')
        x_1 = np.load(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_xgeom_2eff_coef.npy')
        g_1 = np.load(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_constraint_2eff_coef.npy')
        hv_1 = np.load(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_hv_2eff_coef.npy')
        cond_1 = np.load(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_conditions_2eff_coef.npy')
        loss_1 = np.load(f'{save_dir}\\{dt}\\mat_{PM_mat}_temp_{PM_temp}\\{dt}_2DVnabla_losses_2eff_coef.npy')

        check_constraint_last = ~(g_1[-1]>0).sum(axis=1).astype(bool)
        # x_1_all.append(x_1[-1][check_constraint_last])

        PA = f.params_optimization['required_torque_points'][0]
        PB = f.params_optimization['required_torque_points'][1]
        # fig, ax = plt.subplots(figsize=(8/2.54,5/2.54))
        # fig, ax = plt.subplots(figsize=(13/2.54,8/2.54))

        g_1_ = g_1[-1][check_constraint_last]
        alpha = 1.05
        # ax.plot(pf_tmp[:,0]*14.63, pf_tmp[:,1]*-197, c='c', linewidth=1)
        # ax.plot([140,330],[PB[0]*alpha,PB[0]*alpha],'b--', linewidth=1)
        # ax.plot([PA[0]*alpha,PA[0]*alpha],[25,70],'b--', linewidth=1)
        # # ax.plot([140,330],[PB[0],PB[0]],'b--', linewidth=1)
        # # ax.plot([PA[0],PA[0]],[25,70],'b--', linewidth=1)
        # ax.plot((-np.array(g_1)[:,:,0]*PA[0]+PA[0]*alpha).flatten(),
        #         (-np.array(g_1)[:,:,1]*PA[0]+PB[0]*alpha).flatten(),
        #         'o', c='k', ms=1, label='All pop.')
        # ax.plot(-np.array(g_1)[-1,:,0]*PA[0]+PA[0]*alpha,
        #         -np.array(g_1)[-1,:,1]*PA[0]+PB[0]*alpha,
        #         's', c='g', ms=2, label='Last pop.')
        # ax.plot(-np.array(g_1_)[:,0]*PA[0]+PA[0]*alpha,
        #         -np.array(g_1_)[:,1]*PA[0]+PB[0]*alpha,
        #         '^', c='r', ms=2, label='Last pop.')

        # ax.set_xlabel('Torque at P_A (Nm)')
        # ax.set_ylabel('Torque at P_B (Nm)')

        # ax.text(140,40*1.07 ,'Constraint P_B',color='b', fontsize=7)
        # ax.text(197*1.05,26 ,'Constraint P_A',color='b', rotation=-90, fontsize=7)
        # ax.legend(loc='upper left',fontsize=7)

        # fig.tight_layout()

    # for Ie in Ie_list:
        # g_1 = np.load(f'{save_dir}\\Ie_{Ie}\\{dt}_2DVnabla_constraint_2eff_coef.npy')
        # check_constraint_last = ~(g_1[-1]>0).sum(axis=1).astype(bool)
        # loss_1 = np.load(f'{save_dir}\\Ie_{Ie}\\{dt}_2DVnabla_losses_2eff_coef.npy')
        # s = np.array(loss_1).shape

        # fig, ax = plt.subplots(figsize=(12/2.54,8/2.54))
        # ax.plot(np.array(loss_1)[:,:,:,:2].sum(axis=3).reshape(s[0]*s[1], s[2])[:,0],
        #         np.array(loss_1)[:,:,:,:2].sum(axis=3).reshape(s[0]*s[1], s[2])[:,1],
        #         'o', c='k', ms=1, label='All population')
        # ax.plot(np.array(loss_1[-1])[:,:,:2].sum(axis=2).reshape(s[1], s[2])[:,0],
        #         np.array(loss_1[-1])[:,:,:2].sum(axis=2).reshape(s[1], s[2])[:,1],
        #         's', c='g', ms=3, label='Last population')
        loss_1_ = loss_1[-1][:,:,:2].sum(axis=2)
        loss_1_ok = loss_1_[check_constraint_last]
        err = False
        try:
            if check_constraint_last.sum()>4:
                pf_tmp = np.array(calc_pareto_front(loss_1_ok))
            else:
                pf_tmp = loss_1_ok
            pf_tmp_all[f'{PM_mat}_{PM_temp}'] = pf_tmp
            ax.plot(pf_tmp[:,0], pf_tmp[:,1], c='b', linewidth=1)
        except:
            err = True
        # if pf_tmp.shape[0]==0: err = True
        # ax.plot(np.array(loss_1_ok)[:,0],
        #         np.array(loss_1_ok)[:,1],
        #         '^', c='r', ms=3, label='Last population')
        # ax.set_xlabel(f'loss at P_1 (W)')
        # ax.set_ylabel(f'loss at P_2 (W)')

        # ax.legend(loc='best',fontsize=13)
        # # fig.tight_layout()
        # plt.show()

        # if err:
        #     print('no effective solution\n\n')
        # else:
        #     pf_tmp_images = []
        #     pf_tmp_conditions = []
        #     n = -1
        #     res = find_min_m(pf_tmp.shape[0])
        #     fig, axes = plt.subplots(res, res, figsize=(10, 10))
        #     for i in range(pf_tmp.shape[0]):
        #         latent_variable = x_1[n][
        #             np.where(
        #                 # np.array(loss_1_)==pf_tmp[pf_tmp[:,1]<-0.5][i][0]
        #                 np.array(loss_1_)==pf_tmp[i][0]
        #             )[0][0]]
        #         pf_tmp_conditions.append(cond_1[n][
        #             np.where(
        #                 np.array(loss_1_)==pf_tmp[i][0]
        #             )[0][0]]
        #         )
        #     #     im = generate_truncated(GAN.G, torch.from_numpy(latent_variable.reshape(1,-1)).to(device=device, dtype=torch.float) )
        #         im = GAN.G(torch.from_numpy(latent_variable.reshape(1,-1)).to(device=device, dtype=torch.float) )
        #         im_np = im.to('cpu').detach().numpy()[0].transpose(1,2,0)
        #         im_np[im_np>1.]=1.
        #         im_np[im_np<0.]=0.
        #         pf_tmp_images.append(im_np)
        #         # plt.imshow( im_np )
        #         # plt.show()

        #         img_recon = clear_blurred_image(im_np)
        #         img_recon = reconstruct_motor_image(img_recon)
        #         if pf_tmp.shape[0]==1:
        #             axes.imshow(img_recon)
        #             axes.axis('off')
        #         else:
        #             axes.flatten()[i].imshow(img_recon)
        #     if pf_tmp.shape[0]!=1:
        #         for i in range(res**2):
        #             axes.flatten()[i].axis('off')
        #             # plt.imshow(img_recon)
        #             # plt.axis('off')
        #             # plt.show()
        #     plt.show()
    for key in pf_tmp_all.keys():
        if pf_tmp.shape[0]==1:
            plt.plot(pf_tmp_all[key][:,0], pf_tmp_all[key][:,1], 'o', label=key)
        else:
            plt.plot(pf_tmp_all[key][:,0], pf_tmp_all[key][:,1], linewidth=1, label=key)
    plt.legend(loc='best', fontsize=7)
    plt.show()

#%%

#%%

#%%
for key in pf_tmp_all.keys():
    if pf_tmp.shape[0]==1:
        plt.plot(pf_tmp_all[key][:,0], pf_tmp_all[key][:,1], 'o', label=key)
    else:
        plt.plot(pf_tmp_all[key][:,0], pf_tmp_all[key][:,1], linewidth=1, label=key)
# plt.xlim([1250,1700])
plt.legend(loc='upper right', fontsize=7)
plt.show()



#%%
val_x = np.array(x_1)
shape = val_x.shape
res_topology = np.zeros(shape[:2])
for i in tqdm(range(shape[0])):
    for j in range(shape[1]):
#         latent_variable = calculate_latent_variable(val_x[i,j])
        generated_image = GAN.G(torch.from_numpy(val_x[i,j].reshape(1,-1)).to(device=device, dtype=torch.float) )
        image = generated_image.to('cpu').detach().numpy()[0].transpose(1,2,0)
        res_topology[i,j] = judge_topology(image)
# res_topology
#%%
val = np.array(f_1)
val_2D = val[res_topology==3]
val_V = val[res_topology==1]
val_nabra = val[res_topology==2]
val_other = val[np.isnan(res_topology)]

# fig, ax = plt.subplots(figsize=(8/2.54,5/2.54))
fig, ax = plt.subplots(figsize=(12/2.54,8/2.54))

ax.plot(val_nabra.T[0]*-100,val_nabra.T[1]*-100, 'o', c='g', ms=0.8, label='nabra')
ax.plot(val_V.T[0]*-100,val_V.T[1]*-100, 'o', c='b', ms=0.8, label='V')
ax.plot(val_2D.T[0]*-100,val_2D.T[1]*-100, 'o', c='r', ms=0.8, label='2D')
ax.plot(val_other.T[0]*-100,val_other.T[1]*-100, 'o', c='k', ms=0.8, label='other')
# ax.plot([5,17],[197*1.03,197*1.03],'k--', linewidth=1)

# pf_tmp = calc_pareto_front(np.array(f_1[-1]).reshape(s[1], s[2]))
# plt.plot(pf_tmp[:,0]*14.63, pf_tmp[:,1]*-197, c='r', label='pareto front')

# ax.set_yticks(np.arange(120,240+1,40))
# ax.set_yticklabels(np.arange(120,240+1,40))
# ax.set_ylim([120,240])
ax.set_xlabel('Efficiency at PA (%)')
ax.set_ylabel('Efficiency at PB (%)')

# ax.text(5,197*1.05,'Constraint',color='b')
# ax.text(10,205,'A',color='g')
# ax.text(13,222,'B',color='g')
# ax.text(16.5,227,'C',color='g')

ax.legend(loc='best', fontsize=7)

fig.tight_layout()
# plt.savefig(f'{date}_all_population_w_topology_condition1.png', dpi=300, format='png')



#%%
val = np.array(g_1)
val_2D = val[res_topology==3]
val_V = val[res_topology==1]
val_nabra = val[res_topology==2]
val_other = val[np.isnan(res_topology)]

# fig, ax = plt.subplots(figsize=(8/2.54,5/2.54))
fig, ax = plt.subplots(figsize=(12/2.54,8/2.54))

ax.plot(-val_nabra.T[0]*PA[0]+PA[0]*alpha,-val_nabra.T[1]*PA[0]+PB[0]*alpha, 'o', c='g', ms=0.8, label='nabra')
ax.plot(-val_V.T[0]*PA[0]+PA[0]*alpha,-val_V.T[1]*PA[0]+PB[0]*alpha, 'o', c='b', ms=0.8, label='V')
ax.plot(-val_2D.T[0]*PA[0]+PA[0]*alpha,-val_2D.T[1]*PA[0]+PB[0]*alpha, 'o', c='r', ms=0.8, label='2D')
ax.plot(-val_other.T[0]*PA[0]+PA[0]*alpha,-val_other.T[1]*PA[0]+PB[0]*alpha, 'o', c='k', ms=0.8, label='other')


ax.plot([140,220],[PB[0]*alpha,PB[0]*alpha],'b--', linewidth=1)
ax.plot([PA[0]*alpha,PA[0]*alpha],[25,60],'b--', linewidth=1)
# ax.plot((-np.array(g_1)[:,:,0]*PA[0]+PA[0]*alpha).flatten(),
#          (-np.array(g_1)[:,:,1]*PA[0]+PB[0]*alpha).flatten(),
#          'o', c='k', ms=1, label='All pop.')
# ax.plot(-np.array(g_1)[-1,:,0]*PA[0]+PA[0]*alpha,
#          -np.array(g_1)[-1,:,1]*PA[0]+PB[0]*alpha,
#          '^', c='r', ms=2, label='Last pop.')
# ax.plot(-np.array(g_1)[-1,15,0]*PA[0]+PA[0]*alpha,
#          -np.array(g_1)[-1,15,1]*PA[0]+PB[0]*alpha,
#          's', c='g', ms=5)

# pf_tmp = calc_pareto_front(np.array(f_1[-1]).reshape(s[1], s[2]))


# ax.set_yticks(np.arange(120,240+1,40))
# ax.set_yticklabels(np.arange(120,240+1,40))
# ax.set_ylim([120,240])
ax.set_xlabel('Torque at P_A (Nm)')
ax.set_ylabel('Torque at P_B (Nm)')

ax.text(140,40*1.07 ,'Constraint P_B',color='b', fontsize=7)
ax.text(197*1.05,26 ,'Constraint P_A',color='b', rotation=-90, fontsize=7)
# ax.text(10,205,'A',color='g')
# ax.text(13,222,'B',color='g')
# ax.text(16.5,227,'C',color='g')

ax.legend(loc='upper left',fontsize=7)

fig.tight_layout()
# plt.savefig(f'{date}_all_population_w_topology_condition1.png', dpi=300, format='png')


#%%

#%%
params_optimization['verbose'] = False
f = Optimize()
# %timeit f.optimize(Ie_max='134', Vdc='650', torque1='197, 3000', torque2='40, 11000', efficiency1='20, 3500', efficiency2='20, 11000')

#%%

#%%

#%%

#%%
#%%



#%%
class ImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        path_data = Path(params_data['path_data'])
        folder_image = params_data['folder_image']
        fnames = params_data['fnames']
        data_number = params_data['data_number']
        data_image = params_data['data_image']
        data_PM = params_data['data_PM']
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
        }
        self.labels = {}
        for key in data_info.keys():
            self.labels[key] = pd.DataFrame()
        for fname in fnames:
            df = pd.read_csv(path_data/f'{data_number}_{fname}.csv')
            for key, val in data_info.items():
                self.labels[key] = pd.concat((self.labels[key], df[val]), axis=0)
        ## label_pm_material
        key = 'pm_material'
        self.labels[key] = pd.DataFrame()
        for fname in fnames:
            self.labels[key] = pd.concat((
                self.labels[key], pd.read_csv(path_data/f'{data_PM}_{fname}.csv')
            ), axis=0)
        for key in self.labels.keys():
            self.labels[key] = self.labels[key].values
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

        return (
            self.transform(img),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(pm_material, dtype=torch.float32),
            torch.tensor(psi_d, dtype=torch.float32),
            torch.tensor(psi_q, dtype=torch.float32),
            torch.tensor(hysteresis, dtype=torch.float32),
            torch.tensor(joule, dtype=torch.float32),
            torch.tensor(pm_joule, dtype=torch.float32),
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
# indices = range(0, len(dataset), 1000)
indices = range(0, len(valid_dataset), 10)
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
preds_psi_dq_all = np.array(preds_psi_dq_all)
data_psi_dq_all = np.array(data_psi_dq_all)
preds_ironloss_all = np.array(preds_ironloss_all)
data_ironloss_all = np.array(data_ironloss_all)

# %%
preds_all = np.hstack((preds_psi_dq_all,preds_ironloss_all)).T
data_all = np.hstack((data_psi_dq_all,data_ironloss_all)).T
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
print(data_all)
#%%

#%%

#%%
