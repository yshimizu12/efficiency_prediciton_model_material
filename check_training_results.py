#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint

#%%
dir_results = Path("training_results")
header_flux = ["epoch", "train_loss_psi_d", "train_loss_psi_q", "valid_loss_psi_d", "valid_loss_psi_q", "time"]
header_ironloss = ["epoch", "train_loss_hysteresis", "train_loss_joule", "valid_loss_hysteresis", "valid_loss_joule", "time"]
header_all = [header_flux, header_ironloss]

dir_0 = Path("0_base")

dir_base_flux = "2312131429_pred_flux_brhc_swin_t_w_params"
dir_base_ironloss = "2312131429_pred_ironloss_brhc_swin_t_w_params"
dir_base_all = [dir_base_flux, dir_base_ironloss]

name_base = "swin_t"
labels = ['### flux ###','### iron loss ###']

#%%
n_try = 5
df_base_list = []
for dir_base, label, header in zip(dir_base_all, labels, header_all):
    print(label)
    # params.pklを読み込む
    params_file = dir_results / dir_0 / dir_base / "params.pkl"
    params = pd.read_pickle(params_file)
    print(params)
    res_tail = np.array([])
    for i in range(n_try):
        # result.csvを読み込む
        pd_res = pd.read_csv(dir_results / dir_0 / dir_base / f"result_{i}.csv", names=header)
        # display(pd_res.head())
        # display(pd_res.tail(1))
        res_tail = np.append(res_tail, pd_res.tail(1).values)
    res_tail = res_tail.reshape(n_try, -1)[:,1:-1]
    # display(pd.DataFrame(res_tail, columns=header[1:-1]))
    df_base = pd.DataFrame([res_tail.mean(axis=0),res_tail.std(axis=0)], columns=header[1:-1], index=["mean","std"])
    display(df_base)
    df_base_list.append(df_base)
#%%
dir_1 = Path("1_comp_model")
dir_1_list_flux = [p.name for p in dir_results.glob(str(dir_1 / "*")) if p.is_dir() if p.name.split("_")[2]=='flux']
dir_1_list_ironloss = [p.name for p in dir_results.glob(str(dir_1 / "*")) if p.is_dir() if p.name.split("_")[2]=='ironloss']
dir_1_list_all = [dir_1_list_flux, dir_1_list_ironloss]
pprint(dir_1_list_all)

name_1_list_flux = ["_".join(p.name.split("_")[4:-2]) for p in dir_results.glob(str(dir_1 / "*")) if p.is_dir() if p.name.split("_")[2]=='flux']
name_1_list_ironloss = ["_".join(p.name.split("_")[4:-2]) for p in dir_results.glob(str(dir_1 / "*")) if p.is_dir() if p.name.split("_")[2]=='ironloss']
name_1_list_all = [name_1_list_flux, name_1_list_ironloss]
pprint(name_1_list_all)

#%%
df_1_mean_list = []
df_1_std_list = []
for dir_1_list, name_1_list, label, header in zip(dir_1_list_all, name_1_list_all, labels, header_all):
    print(label)
    mean_list = []
    std_list = []
    for dir_, name_ in zip(dir_1_list, name_1_list):
        # params.pklを読み込む
        print(name_)
        params_file = dir_results / dir_1 / dir_ / "params.pkl"
        params = pd.read_pickle(params_file)
        print(params)
        res_tail = np.array([])
        n_try = 5 if name_ != "vit_l_16" else 2
        for i in range(n_try):
            # result.csvを読み込む
            pd_res = pd.read_csv(dir_results / dir_1 / dir_ / f"result_{i}.csv", names=header)
            # display(pd_res.head())
            # display(pd_res.tail(1))
            res_tail = np.append(res_tail, pd_res.tail(1).values)
        res_tail = res_tail.reshape(n_try, -1)[:,1:-1]
        mean_list.append(res_tail.mean(axis=0))
        std_list.append(res_tail.std(axis=0))
        # display(pd.DataFrame(res_tail, columns=header[1:-1]))
    df_mean = pd.DataFrame(mean_list, columns=header[1:-1], index=name_1_list)
    df_std = pd.DataFrame(std_list, columns=header[1:-1], index=name_1_list)
    display(df_mean)
    display(df_std)
    df_1_mean_list.append(df_mean)
    df_1_std_list.append(df_std)

# %%
cols_list = [["valid_loss_psi_d", "valid_loss_psi_q"], ["valid_loss_hysteresis", "valid_loss_joule"]]
for df_mean, df_std, df_base, cols, label, header in zip(df_1_mean_list, df_1_std_list, df_base_list, cols_list, labels, header_all):
    print(label)
    display(df_mean)
    for col in cols:
        plt.figure()
        plt.bar(
            np.append(df_mean.index.values, name_base),
            np.append(df_mean[col].values, df_base[col]["mean"]),
            yerr=np.append(df_std[col].values, df_base[col]["std"]),
            capsize=5
        )
        plt.ylabel("Valid Loss (-)")
        plt.title(col)
        plt.show()

#%%


#%%

#%%

#%5