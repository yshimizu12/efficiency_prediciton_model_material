#%%
import numpy as np
import pandas as pd
from pathlib import Path

#%%
dir_results = Path("training_results")
header_flux = ["epoch", "train_loss_psi_d", "train_loss_psi_q", "valid_loss_psi_d", "valid_loss_psi_q", "time"]
header_ironloss = ["epoch", "train_loss_hysteresis", "train_loss_joule", "valid_loss_hysteresis", "valid_loss_joule", "time"]
header_all = [header_flux, header_ironloss]

dir_base_flux = "2312121413_pred_flux_brhc_swin_t_w_params"
dir_base_ironloss = "2312111811_pred_ironloss_brhc_swin_t_w_params"
dir_base_all = [dir_base_flux, dir_base_ironloss]

labels = ['### flux ###','### iron loss ###']
#%%
for dir_base, label, header in zip(dir_base_all, labels, header_all):
    print(label)
    # params.pklを読み込む
    params_file = dir_results / dir_base / "params.pkl"
    params = pd.read_pickle(params_file)
    print(params)
    # result.csvを読み込む
    pd_res = pd.read_csv(dir_results / dir_base / "result_0.csv", names=header)
    display(pd_res.head())




