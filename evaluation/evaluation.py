import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.factory import get_sampling, get_termination
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
# from pymoo.factory import get_performance_indicator
from pymoo.operators.sampling.lhs import LHS

class Evaluate:
    def __init__(
        self, regression_model_motorparameter, regression_model_ironloss,
        Iem=134, Nmax=20000, TEMP_PM=20, PM_material='NMX-S49CH', Vdc=650, Ra=0.1285, Pn=4, device='cpu', param_scaling=None,
        **kwargs,
        ):
        self.regression_model_motorparameter = regression_model_motorparameter
        self.regression_model_ironloss = regression_model_ironloss
        self.regression_model_motorparameter.eval()
        self.regression_model_ironloss.eval()

        self.Ra = Ra
        self.Pn = Pn
        self.Nmax = Nmax
        self.N_interval = 1000 if self.Nmax >= 10000 else self.Nmax//10
        self.TEMP_PM = TEMP_PM
        self.device = device
        # self.sp = sp
        self.param_scaling = param_scaling
        self._elec_params_setting(Iem, Vdc)
        self._init_envs_matplotlib()

    def evaluation(self, img, filename):
        img = np.array(img)
        rotor_image_tensor = torch.from_numpy(np.array([
            img.transpose(2,0,1).astype(np.float32)
            ])).clone().to(self.device)
        coefs = self.calc_coef(rotor_image_tensor)
        df_NT_pred_all, _, _ = self._calc_NT(coefs, rotor_image_tensor, Ns=range(self.N_interval,self.Nmax+self.N_interval,self.N_interval), Ie=self.Iem)
        self._create_efficiency_map(
            df_NT_pred_all,
            charac_num=0,
            image_save=True,
            name=filename,
            pltshow=False,
            create_plot=True,
            )

    def calc_coef(self, image, calc_grad=False):
        coefs = self.regression_model_motorparameter(image)
        # sp = [sp_.copy() for sp_ in self.sp]
        # if calc_grad:
            # sp = [torch.from_numpy(sp_).to(device=self.device,dtype=torch.float) for sp_ in sp]
        # else:
            # coefs = [coefs[i].to('cpu').detach().numpy().copy() for i in range(3)]
        # coefs = [c[0] * sp_[1] + sp_[0] for sp_, c in zip(sp, coefs)]
        return coefs
    
    def create_efficiency_map(self, rotor_image_tensor):
        coefs = self.calc_coef(rotor_image_tensor)
        df_NT_pred_all, _, _ = self._calc_NT(coefs, rotor_image_tensor, Ns=range(self.N_interval,self.Nmax+self.N_interval,self.N_interval), Ie=self.Iem)
        params_plot = self._create_efficiency_map(
            df_NT_pred_all,
            charac_num=0,
            image_save=False,
            pltshow=False,
            create_plot=False,
        )
        return params_plot, df_NT_pred_all['Tavg (Nm)'].values[0]

    def _sin(self, theta): return np.sin(np.radians(theta))
    def _cos(self, theta): return np.cos(np.radians(theta))
    # def _calculate_magnet_volume(self, image):
    #     return float((image.max(axis=1)[1]==2).sum())/(256*256)*100

    def _calc_current_P(self, current):
        current = np.array(current)
        return np.stack((
            current**3,
            current**2,
            current,
            np.ones(current.shape),
            ))

    def _calc_current_Ld(self, id_L, iq_L):
        id_L = np.array(id_L)
        iq_L = np.array(iq_L)
        return np.stack((
            np.ones(id_L.shape),
            id_L,
            iq_L,
            id_L**2,
            id_L * iq_L,
            iq_L**2,
            id_L**3,
            (id_L**2) * iq_L,
            id_L * (iq_L**2),
            iq_L**3,
            ))

    def _calc_current_Lq(self, id_L, iq_L):
        id_L = np.array(id_L)
        iq_L = np.array(iq_L)
        return np.stack((
            np.ones(id_L.shape),
            id_L,
            iq_L,
            id_L**2,
            id_L * iq_L,
            iq_L**2,
            id_L**3,
            (id_L**2) * iq_L,
            id_L * (iq_L**2),
            iq_L**3,
            id_L**4,
            (id_L**3) * iq_L,
            (id_L**2) * (iq_L**2),
            id_L * (iq_L**3),
            iq_L**4,
            ))

    def _motor_parameter_calculation(self, Ia, beta, coefs):
        Ia, beta = np.array([Ia]).reshape(-1), np.array([beta]).reshape(-1)
        assert beta.shape[0] == Ia.shape[0], "Ia and beta should be the same shape!"
        id_ = -Ia*self._sin(beta)
        iq_ = Ia*self._cos(beta)

        current_P = self._calc_current_P(Ia)
        current_Ld = self._calc_current_Ld(id_, iq_)
        current_Lq = self._calc_current_Lq(id_, iq_)
        P_pred = np.dot(coefs[0], current_P)
        Ld_pred = np.dot(coefs[1], current_Ld)
        Lq_pred = np.dot(coefs[2], current_Lq)
        return P_pred, Ld_pred, Lq_pred

    def _motor_parameter_calculation_grad(self, Ia, beta, coefs):
        id_ = -Ia*self._sin(beta)
        iq_ = Ia*self._cos(beta)
        #.reshape(1,-1)
        current_P = torch.from_numpy(self._calc_current_P(Ia)).to(device=self.device,dtype=torch.float)
        current_Ld = torch.from_numpy(self._calc_current_Ld(id_, iq_)).to(device=self.device,dtype=torch.float)
        current_Lq = torch.from_numpy(self._calc_current_Lq(id_, iq_)).to(device=self.device,dtype=torch.float)
        P_pred = torch.matmul(coefs[0], current_P)
        Ld_pred = torch.matmul(coefs[1], current_Ld)
        Lq_pred = torch.matmul(coefs[2], current_Lq)
        return P_pred, Ld_pred, Lq_pred

    def _torque_calculation(self, Ia, beta, coefs, calc_grad=False):
        P_pred, Ld_pred, Lq_pred = self._motor_parameter_calculation_grad(Ia, beta, coefs) if calc_grad else self._motor_parameter_calculation(Ia, beta, coefs)
        Tmag = self.Pn*Ia*P_pred*self._cos(beta)
        Trel = self.Pn*0.5*(Ia**2)*(Lq_pred-Ld_pred)*self._sin(2*beta)
        return Tmag+Trel

    def _speed_limit_calculation(self, Ia, beta, coefs):
        P_pred, Ld_pred, Lq_pred = self._motor_parameter_calculation(Ia, beta, coefs)
        Vom = self.Vam-self.Ra*Ia
        Psi_o = np.sqrt( ( P_pred-Ld_pred*Ia*self._sin(beta) )**2+( Lq_pred*Ia*self._cos(beta) )**2 )
        return Vom/Psi_o*60/(2*np.pi)/self.Pn

    def _mtpa_beta_search(self, coefs, minbeta=0, maxbeta=90, interval=5):
        betas = np.arange(minbeta, maxbeta, interval)
        Ia = np.ones(betas.shape[0])*self.Iam
        Tcal = self._torque_calculation(Ia, betas, coefs)
        index_max = Tcal.argmax()
        beta_max = betas[index_max]

        interval_detail = interval*0.02
        betas_detail = np.arange(
            beta_max-interval+interval_detail,
            beta_max+interval,
            interval_detail)
        Ia_detail = np.ones(betas_detail.shape[0])*self.Iam
        T_detail = self._torque_calculation(Ia_detail, betas_detail, coefs)
        index_max = T_detail.argmax()
        return betas_detail[index_max], T_detail[index_max]

    def _mesh_grid(self, x_min, x_max, x_interval, y_min, y_max, y_interval):
        x = np.arange(x_max, x_min, x_interval)
        y = np.append(np.arange(y_min, y_max, y_interval), y_max)
        num_x, num_y = x.shape[0], y.shape[0]
        xx, yy = np.meshgrid(x, y)
        return xx, yy, num_x, num_y

    def _drivable_minimum_judge(self, array, judge_array):
        num = judge_array.shape[1]
        judge_array_edge = judge_array*np.vstack((judge_array[0],np.logical_not(judge_array)[:-1,:])).astype(np.bool)
        return array[judge_array_edge]

    def _mtpv_search(self, coefs, minbeta, maxbeta=90, interval_beta=5,  Nx=3000):
        interval_Ia = -1. if self.Iam >= 50. else -self.Iam*0.02
        Ias, betas, num_Ias, num_betas = self._mesh_grid(0, self.Iam, interval_Ia, minbeta, maxbeta, interval_beta)
        N_limit = self._speed_limit_calculation(Ias.flatten(), betas.flatten(), coefs)
        N_judge = N_limit.reshape(num_betas, num_Ias) >= Nx
        assert N_judge[0][0] == 0, "MTPA control available!"
        if np.count_nonzero(N_judge) != 0:
            Ias_drivable = self._drivable_minimum_judge(Ias, N_judge)
            betas_drivable = self._drivable_minimum_judge(betas, N_judge)

            num_step = 10
            range_for_beta = range(num_step)[::-1]
            Ias_detail = np.tile(Ias_drivable, (num_step, 1))
            betas_detail = np.array([betas_drivable-interval_beta*i/num_step for i in range_for_beta])
            N_limit_detail =  self._speed_limit_calculation(Ias_detail.flatten(), betas_detail.flatten(), coefs)
            N_judge_detail = N_limit_detail.reshape(num_step, betas_drivable.shape[0]) >= Nx
            Ias_drivable_detail =  self._drivable_minimum_judge(Ias_detail, N_judge_detail)
            betas_drivable_detail =  self._drivable_minimum_judge(betas_detail, N_judge_detail)
            N_drivable_detail =  self._drivable_minimum_judge(N_limit_detail.reshape(num_step, betas_drivable.shape[0]),N_judge_detail)
            Ts_drivable_detail =  self._torque_calculation(Ias_drivable_detail, betas_drivable_detail, coefs)
            index = Ts_drivable_detail.argmax()
            Ia_MTPV = Ias_drivable_detail[index]
            beta_MTPV = betas_drivable_detail[index]
            N_MTPV = N_drivable_detail[index]
            T_MTPV = Ts_drivable_detail[index]
            return Ia_MTPV, beta_MTPV, N_MTPV, T_MTPV
        else: return np.nan, np.nan, np.nan, np.nan

    def _search(self, Nx, coefs):
        beta_MTPA, T_MTPA = self._mtpa_beta_search(coefs, 0, 90, 5)
        Nbase = float(self._speed_limit_calculation(self.Iam, beta_MTPA, coefs))
        if Nx <= Nbase: return self.Iam, beta_MTPA, Nx, T_MTPA
        else:
            return self._mtpv_search(coefs, beta_MTPA, 90, 5, Nx)

    def _mtpa_beta_search_other_current(self, Iam_tmp, coefs, minbeta=0, maxbeta=90, interval=5):
        betas = np.arange(minbeta, maxbeta, interval)
        Ia = np.ones(betas.shape[0])*Iam_tmp
        Tcal = self._torque_calculation(Ia, betas, coefs)
        index_max = Tcal.argmax()
        beta_max = betas[index_max]

        interval_detail = interval*0.02
        betas_detail = np.arange(
            beta_max-interval+interval_detail,
            beta_max+interval,
            interval_detail
            )
        Ia_detail = np.ones(betas_detail.shape[0])*Iam_tmp
        T_detail = self._torque_calculation(Ia_detail, betas_detail, coefs)
        index_max = T_detail.argmax()
        return betas_detail[index_max], T_detail[index_max]

    def _mtpv_search_other_current(self, Iam_tmp, coefs, minbeta, maxbeta=90, interval_beta=5,  Nx=3000):
        interval_Ia = -1. if Iam_tmp >= 50. else -Iam_tmp*0.02
        Ias, betas, num_Ias, num_betas = self._mesh_grid(0, Iam_tmp, interval_Ia, minbeta, maxbeta, interval_beta)
        N_limit = self._speed_limit_calculation(Ias.flatten(), betas.flatten(), coefs)
        N_judge = N_limit.reshape(num_betas, num_Ias) >= Nx
        assert N_judge[0][0] == 0, "MTPA control available!"
        if np.count_nonzero(N_judge) != 0:
            Ias_drivable = self._drivable_minimum_judge(Ias, N_judge)
            betas_drivable = self._drivable_minimum_judge(betas, N_judge)

            num_step = 10
            range_for_beta = range(num_step)[::-1]
            Ias_detail = np.tile(Ias_drivable, (num_step, 1))
            betas_detail = np.array([betas_drivable-interval_beta*i/num_step for i in range_for_beta])
            N_limit_detail =  self._speed_limit_calculation(Ias_detail.flatten(), betas_detail.flatten(), coefs)
            N_judge_detail = N_limit_detail.reshape(num_step, betas_drivable.shape[0]) >= Nx
            Ias_drivable_detail =  self._drivable_minimum_judge(Ias_detail, N_judge_detail)
            betas_drivable_detail =  self._drivable_minimum_judge(betas_detail, N_judge_detail)
            N_drivable_detail =  self._drivable_minimum_judge(N_limit_detail.reshape(num_step, betas_drivable.shape[0]), N_judge_detail)
            Ts_drivable_detail =  self._torque_calculation(Ias_drivable_detail, betas_drivable_detail, coefs)
            index = Ts_drivable_detail.argmax()
            Ia_MTPV = Ias_drivable_detail[index]
            beta_MTPV = betas_drivable_detail[index]
            N_MTPV = N_drivable_detail[index]
            T_MTPV = Ts_drivable_detail[index]
            return Ia_MTPV, beta_MTPV, N_MTPV, T_MTPV
        else: return np.nan, np.nan, np.nan, np.nan

    def _search_other_current(self, Nx, coefs, Iam_tmp):
        beta_MTPA, T_MTPA = self._mtpa_beta_search_other_current(Iam_tmp, coefs, 0, 90, 5)
        Nbase = float(self._speed_limit_calculation(Iam_tmp, beta_MTPA, coefs))
        if Nx <= Nbase: return Iam_tmp, beta_MTPA, Nx, T_MTPA #MTPA制御可能
        else:
            return self._mtpv_search_other_current(Iam_tmp, coefs, beta_MTPA, 90, 5, Nx)

    def _search_current_condition_for_maximum_torque_control(self, Nx, torque, coefs, Iam_tmp):
        Ia_MT, beta_MT, N_MT, T_MT = self._search_other_current(Nx, coefs, Iam_tmp)
        if T_MT < torque: return (None,None,None,None)
        else:
            interval_Ia = -10. if Iam_tmp >= 100. else -Iam_tmp*0.1
            Ias = np.arange(Iam_tmp+interval_Ia, 0, interval_Ia)
            charac_arr = [self._search_other_current(Nx, coefs, Ia) for Ia in Ias] 
            charac_arr.insert(0,(Ia_MT, beta_MT, N_MT, T_MT))
            charac_arr = np.array(charac_arr)
            ind = abs(charac_arr[:,3]-torque).argsort()[:2]
#             print(charac_arr[:,0])
            Ias_detail = np.arange(charac_arr[:,0][ind[0]],charac_arr[:,0][ind[1]],
                                   np.sign(charac_arr[:,0][ind[0]]-charac_arr[:,0][ind[1]])*interval_Ia/20)
            charac_arr_detail = [self._search_other_current(Nx, coefs, Ia) for Ia in Ias_detail]
            charac_arr_detail.insert(0,charac_arr[ind[1]])
            charac_arr_detail = np.array(charac_arr_detail)
            return charac_arr_detail[abs(charac_arr_detail[:,3]-torque).argmin()]

    def _scaling_ironloss(self, id_, iq_):
        id_scaled = (id_-self.param_scaling['id']['mean'])/self.param_scaling['id']['std']
        iq_scaled = (iq_-self.param_scaling['iq']['mean'])/self.param_scaling['iq']['std']
        return id_scaled, iq_scaled

    def _scaling_speed(self, N):
        N_scaled = (N-self.param_scaling['N']['mean'])/self.param_scaling['N']['std']
        return N_scaled

    def _ironloss_calculation(self, id_, iq_, N, image, calc_grad=False):
        id_, iq_ = self._scaling_ironloss(id_, iq_)
        N = self._scaling_speed(N)
        jou_pred_scaled, hys_pred_scaled = self.regression_model_ironloss(
            image,
            torch.reshape(torch.tensor([id_, iq_], dtype=torch.float32), (1, 2)).to(self.device),
            torch.reshape(torch.tensor([N], dtype=torch.float32), (1, 1)).to(self.device),
        )
        jou_pred = jou_pred_scaled * self.param_scaling['joule']['std'] + self.param_scaling['joule']['mean']
        hys_pred = hys_pred_scaled * self.param_scaling['hysteresis']['std'] + self.param_scaling['hysteresis']['mean']
        return (hys_pred+jou_pred).squeeze() if calc_grad else float(hys_pred+jou_pred)

    def _loss_calculation(self, Ia, beta, N, image, calc_grad=False):
        try:
            copper_loss = self.Ra*Ia**2
            id_ = -Ia*self._sin(beta)
            iq_ = Ia*self._cos(beta)
            iron_loss = self._ironloss_calculation(id_, iq_, N, image, calc_grad)
        except:
            print('error')
            copper_loss, iron_loss = None, None
        return copper_loss, iron_loss

    def _calc_NT(self, coefs, rotor_image_tensor, Ns=range(1000,14000+1,1000), Ie=134):
        Ia_range = np.arange(Ie,0,-15)
        Ia_range = Ia_range*3**0.5
        NT_pred_all = np.array([[self._search_other_current(N, coefs, Ia) for N in Ns] for Ia in Ia_range])
        NT_pred_all_tmp = NT_pred_all.copy()
        beta_T_base = np.array([self._mtpa_beta_search_other_current(Ia, coefs, 0, 90, 5) for Ia in Ia_range])
        N_base = np.array([self._speed_limit_calculation(
            Ia,
            beta_base,
            coefs
        )[0] for Ia, beta_base in zip(Ia_range, beta_T_base[:,0])])
        base_all = np.vstack((np.vstack((np.vstack((Ia_range,beta_T_base[:,0])),N_base)),beta_T_base[:,1])).T

        NT_pred_all = np.array([
            np.insert(NT_pred_all[i],int(base_all[i,2]/1000),base_all[i],axis=0) for i in range(NT_pred_all.shape[0])
        ])

        s_tmp = NT_pred_all.shape
        NT_pred_all = NT_pred_all.reshape(s_tmp[0]*s_tmp[1],s_tmp[2])
        NT_pred_all = NT_pred_all[~np.isnan(NT_pred_all).any(axis=1), :]
        NT_pred_all = np.vstack((NT_pred_all,[[0, 0, N, 0] for N in Ns]))

        loss_pred_all = np.array([self._loss_calculation(Ia, beta, N, rotor_image_tensor)
                                for Ia, beta, N in zip(NT_pred_all[:,0],NT_pred_all[:,1],NT_pred_all[:,2])])
        power_pred = NT_pred_all[:,3] * 2*np.pi/60 * NT_pred_all[:,2]
        efficiency_pred = (power_pred[:-14]-loss_pred_all.T[1][:-14])/(power_pred[:-14]+loss_pred_all.T[0][:-14])
        mat = np.zeros([14])
        mat[:] = np.nan
        efficiency_pred = np.hstack((efficiency_pred, mat))

        df_NT_pred_all = pd.DataFrame(np.vstack((np.vstack((NT_pred_all[:,2:].T,
                                                            np.vstack((loss_pred_all.T,loss_pred_all.sum(axis=1)))
                                                            )),
                                                efficiency_pred*100)).T,
                                    columns=['N (min-1)','Tavg (Nm)','Wc (W)','Wi (W)','W (W)','Efficiency (%)'],)

        NT_pred_all_tmp = np.vstack((NT_pred_all_tmp,[[[0, 0, N, 0] for N in Ns]]))
        loss_pred_all = np.array([[self._loss_calculation(Ia, beta, N, rotor_image_tensor)
                                for Ia, beta, N in zip(NT_pred_all_tmp[i,:,0],NT_pred_all_tmp[i,:,1],NT_pred_all_tmp[i,:,2])]
                                for i in range(NT_pred_all_tmp.shape[0])])
        pred_all = np.concatenate((NT_pred_all_tmp[:,:,2:],loss_pred_all),2)

        # N_min = 0
        # N_max = 14000
        # N_arr_tmp = np.arange(N_min+1000,N_max+1000,1000)
        # df_pred_all_list = []
        # n_Ia = pred_all.shape[0]
        # for i in range(n_Ia):
        #     df_tmp = pd.DataFrame(pred_all[i],columns=['N (min-1)','Tavg (Nm)','Wc (W)','Wi (W)'])
        #     df_tmp['N (min-1)'] = Ns #N_arr_tmp
        #     df_pred_all_list.append(df_tmp)
        # return df_NT_pred_all, df_pred_all_list, n_Ia
        return df_NT_pred_all, None, None

    def _elec_params_setting(self, Iem, Vdc):
        self.Iem = Iem
        self.Iam = self.Iem*np.sqrt(3)
        self.Vdc = Vdc
        self.Vam = np.sqrt(3/2)*4/np.pi*self.Vdc/2

    def _init_envs_matplotlib(self):
        # plt.rcParams['axes.grid'] = True
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

    def _create_efficiency_map(
        self,
        df,
        charac_num=0,
        image_save=False,
        name='init',
        pltshow=True,
        N_interval=1000.,
        N_min=1000,
        N_num=15,
        xlim=[1000,14000],
        xticks=np.hstack((np.arange(1000,14000,5000),[14000])),
        xlabel='Speed (min$^{-1}$)',
        ylabel='Torque (Nm)',
        ylim=[20,250],
        yticks=np.hstack(([20],np.arange(50,250+1,50))),
        contour_levels=[70,80,85,90,95,96,97,98,100],
        create_plot=False,
        ):
        if charac_num == 0:
            characteristics = 'Efficiency (%)'
            cmap = 'jet'
            levels = contour_levels
        elif charac_num == 1:
            characteristics = 'Wc (W)'
            cmap = 'Reds'
            levels = None
        elif charac_num == 2:
            characteristics = 'Wi (W)'
            cmap = 'Blues'
            levels = None
        elif charac_num == 3:
            characteristics = 'W (W)'
            cmap = 'Greens'
            levels = None
        else:
            raise ValueError("charac should be 0,1,2,or3!")

        indices = df.index[df['N (min-1)']==N_min]
        ind_dif = [indices[j+1]-indices[j] for j in range(len(indices)-1)]
        Ns = []
        Ts = []
        characs = []
        for j in range(len(indices)-1):
            N = np.array(df['N (min-1)'].iloc[indices[j]:indices[j+1]])
            T = np.array(df['Tavg (Nm)'].iloc[indices[j]:indices[j+1]])
            charac = np.array(df[characteristics].iloc[indices[j]:indices[j+1]])
            if ind_dif[j] != N_num:
                N_tmp = np.arange(0., 14001.,N_interval)
                N_tmp[:ind_dif[j]] = N
                N = N_tmp
                T_tmp = np.empty(N_num)
                T_tmp[:] = np.nan
                T_tmp[:ind_dif[j]] = T
                T = T_tmp
                charac_tmp = np.empty(N_num)
                charac_tmp[:] = np.nan
                charac_tmp[:ind_dif[j]] = charac
                charac = charac_tmp
            Ns.append(N)
            Ts.append(T)
            characs.append(charac)
        params_plot = [
            Ns, Ts, characs, cmap, levels, xlim, xticks, ylim, yticks, xlabel, ylabel, characteristics
        ]
        if create_plot:
            fig = plt.figure(figsize=(12/2.54,8/2.54))
            ax = fig.add_subplot(111)
            self.create_plot_effciency_map(fig, ax, *params_plot)
            fig.tight_layout()
            # plt.close(fig)
            if image_save:
                plt.savefig(name, dpi=300)
            if pltshow:
                plt.show()
        else:
            return params_plot

    def create_plot_effciency_map(
        self, fig, ax, Ns, Ts, characs, cmap, levels, 
        xlim, xticks, ylim, yticks, xlabel, ylabel, characteristics,
        points=None, values=None,
        ):
        ax.plot(Ns[0],Ts[0],'k-')
        cont = ax.contourf(Ns, Ts, characs, cmap=cmap, levels=levels)

        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if points is not None:
            for point in points:
                ax.plot(point[1], point[0], 'kx', ms=10)
            if values is not None:
                for point, value in zip(points, values[:-1]): 
                    ax.plot(point[1], point[0], 'kx', ms=10)
                    ax.text(point[1]+500, point[0]+5, '{:.1f}%'.format(value))
                ax.text(points[len(values)-1][1]+100, values[-1]+10, '{:.1f}'.format(values[-1]))
                
        fig.colorbar(cont, ax=ax, orientation="vertical", label=characteristics)

class Optimize:
    def __init__(
        self,
        params_prediction, 
        params_optimization,
        ):
        self.params_optimization = params_optimization
        self.problem = MyProblem(
            params_prediction,
            **params_optimization,
        )
        # sampling = get_sampling('real_lhs')
        sampling = LHS()
        self.algorithm = NSGA2(
            pop_size=params_optimization['pop_size'], 
            sampling=sampling,
            n_offsprings=params_optimization['n_offsprings'],
            eliminate_duplicates=True
        )
        self.termination = get_termination("n_gen", params_optimization['n_termination'])

    def optimize(self,seed=0):
        t1 = time.time()
        res = minimize(
            self.problem,
            self.algorithm,
            termination=self.termination,
            seed=seed,
            callback=MyCallbackHV(),
            save_history=False,
            verbose=self.params_optimization['verbose'],
        )
        print(f'elapsed time: {time.time()-t1} sec')
        self.result =  res.algorithm.callback.data
    
    def show_best_result(self,):
        if self.result is None:
            print('Optimize first!')
        else:
            self.best_X = self.result['X'][-1][self.result['F'][-1].sum(axis=1).argmin()]
            self.generated_image_best = self.problem.Generator(torch.from_numpy(self.best_X.reshape(1,-1)).to(device=self.problem.device,dtype=torch.float))
            self._show_result(self.generated_image_best)
            self._calc_current_condition_best()
            self._calc_gradient_best()

    # def show_different_result(self, Iem, Vdc, gain_grad):
    def show_different_result(self, Iem=134, Vdc=650, gain1=0., gain2=0., gain3=0.):
        gain_grad = np.array([gain1, gain2, gain3])
        if np.nonzero(gain_grad)[0].shape[0]>0:
            gain_grad = torch.from_numpy(gain_grad).to(device=self.problem.device,dtype=torch.float)
            X = torch.matmul(gain_grad,self.gradient_best).reshape(1,-1)
            # print(X.sum())
            X += torch.from_numpy(self.best_X.reshape(1,-1)).to(device=self.problem.device,dtype=torch.float)
            self._show_different_result(Iem, Vdc, X)
        else:
            self._show_different_result(Iem, Vdc)

    def _show_result(self, generated_image, calc_value=False):
        params_plot, Tmax = self.problem.eval.create_efficiency_map(generated_image)
        generated_image_ = generated_image.cpu().detach().numpy()[0].transpose(1,2,0)
        generated_image_[generated_image_>1.] = 1.
        generated_image_[generated_image_<0.] = 0.

        values=None
        if calc_value:
            values = []
            for cond in self.condition_evaluation_point_best:
                Ia, beta, N, T = cond['Ia'], cond['beta'], cond['N'], cond['T']
                loss = self.problem.eval._loss_calculation(Ia, beta, N, generated_image, calc_grad=False)
                power = N * 2*np.pi/60 * T
                efficiency = (power-loss[1])/(power+loss[0])*100
                values.append(efficiency)
            values.append(Tmax)

        fig, ax = plt.subplots(1,2,figsize=(10,3))
        plt.subplots_adjust(wspace=0.1, hspace=0.6)
        ax[0].imshow( generated_image_ )
        ax[0].axis('off')
        # fig, ax = None, [None,None]
        self.problem.eval.create_plot_effciency_map(
            fig, ax[1], *params_plot, 
            points=np.vstack((self.problem.evaluate_efficiency_points,self.problem.required_torque_points)),
            values=values
            )
        # plt.imshow( generated_image_ )
        # plt.axis('off')
        # plt.show()
        # fig, ax = plt.subplots(1,1,figsize=(5,3))
        # self.problem.eval.create_plot_effciency_map(fig, ax, *params_plot)
        plt.show()

    def _calc_current_condition_best(self, ):
        if self.generated_image_best is None:
            print('Optimize first!')
        else:
            coefs = self.problem.eval.calc_coef(self.generated_image_best)
            self.Iam_best = self.problem.eval.Iam
            self.beta_mtpa_best, _ = self.problem.eval._mtpa_beta_search(coefs, 0, 90, 5)
            self.condition_evaluation_point_best = []
            for evaluate_point in self.problem.evaluate_efficiency_points:
                Ia_point, beta_point, N_point, T_point = \
                    self.problem.eval._search_current_condition_for_maximum_torque_control(
                        evaluate_point[1], evaluate_point[0], coefs, self.Iam_best)
                self.condition_evaluation_point_best.append(
                    {'Ia': Ia_point, 'beta': beta_point, 'N': N_point, 'T': T_point}
                )

    def _calc_gradient_best(self,):
        if self.generated_image_best is None:
            print('Optimize first!')
        else:
            x_tensor = torch.from_numpy(self.best_X.reshape(1,-1)).to(device=self.problem.device,dtype=torch.float)
            gradient = self._calculate_gradient(
                    x_tensor=x_tensor, charac='torque', Ia=self.Iam_best, beta=self.beta_mtpa_best
            )
            self.gradient_best = gradient / torch.norm(gradient)
            for cond in self.condition_evaluation_point_best:
                gradient = self._calculate_gradient(
                    x_tensor=x_tensor, charac='efficiency', 
                    Ia=cond['Ia'], beta=cond['beta'], N=cond['N'], T=cond['T']
                )
                self.gradient_best = torch.vstack((self.gradient_best, gradient / torch.norm(gradient)))
            print('completed!')

    def _calculate_gradient(
        self, x_tensor, charac=None, 
        Ia=None, beta=None, N=None, T=None
        ):
        x_tensor.requires_grad = True
        generated_image = self.problem.Generator(x_tensor)
        if charac=='efficiency':
            loss = self.problem.eval._loss_calculation(Ia, beta, N, generated_image, calc_grad=True)
            power = N * 2*np.pi/60 * T
            efficiency = (power-loss[1])/(power+loss[0])
            efficiency.backward()
        elif charac=='torque':
            coefs = self.problem.eval.calc_coef(generated_image, calc_grad=True)
            torque = self.problem.eval._torque_calculation(Ia, beta, coefs, calc_grad=True)
            torque.backward()
        else:
            print('select charac from ["efficiency", "torque"]')
        return x_tensor.grad + 1e-16

    def _show_different_result(self, Iem, Vdc, X=None):
        self.problem.eval._elec_params_setting(Iem, Vdc)
        if X is None:
            self._show_result(self.generated_image_best, calc_value=True)
        else:
            generated_image = self.problem.Generator(X)
            self._show_result(generated_image, calc_value=True)


class MyProblem(ElementwiseProblem):
    def __init__(
        self, params_prediction, required_torque_points=None, evaluate_efficiency_points=None, n_var=256, n_obj=2, n_constr=2, xl=np.ones(256)*-1000, xu=np.ones(256)*1000,
        GAN=None, regression_model_motorparameter=None, regression_model_ironloss=None, **kwargs,
    ):
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu,
            # elementwise_evaluation=True,
        )
        self.eval = Evaluate(
            regression_model_motorparameter=regression_model_motorparameter,
            regression_model_ironloss=regression_model_ironloss,
            **params_prediction)
        self.device = params_prediction['device']
        self.dimension = self.n_var        
        self.Generator = GAN.G
        self.latent_dim = self.n_var
        self.regression_model = regression_model_motorparameter
        self.regression_model_ironloss = regression_model_ironloss
        self.required_torque_points = required_torque_points
        self.num_of_req_points = self.required_torque_points.shape[0]
        self.evaluate_efficiency_points = evaluate_efficiency_points

    def _evaluate(self, x, out, *args, **kwargs):
        generated_image = self.Generator(torch.from_numpy(x.reshape(1,-1)).to(device=self.device,dtype=torch.float))
        coefs = self.eval.calc_coef(generated_image)
        # torque
        charac_at_req = np.array([ self.eval._search(N_req, coefs) for N_req in self.required_torque_points[:,1] ])
        # efficiency
        efficiencies = []
        conditions = []
        losses = []
        for evaluate_point in self.evaluate_efficiency_points:
            try:
                Ia_point, beta_point, N_point, T_point = self.eval._search_current_condition_for_maximum_torque_control(evaluate_point[1], evaluate_point[0], coefs, self.eval.Iam)
                loss = self.eval._loss_calculation(Ia_point, beta_point, evaluate_point[1], generated_image)
                power = N_point * 2*np.pi/60 * T_point
                efficiencies.append( -(power-loss[1])/(power+loss[0]) )
                conditions.append([Ia_point, beta_point, N_point, T_point])
                losses.append(loss+(power,))
            except:
                efficiencies.append( np.nan )
                conditions.append([np.nan, np.nan, np.nan, np.nan])
                losses.append([np.nan, np.nan, np.nan])
        conditions = np.array(conditions)
        losses = np.array(losses)
        out["F"] = efficiencies
        out["G"] = [(self.required_torque_points[i][0]*1.05-charac_at_req[i,3])/self.required_torque_points[0][0] for i in range(self.num_of_req_points)]
        out["COND"] = conditions #conditions
        out["LOSS"] = losses

class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["F"] = []
        self.data["X"] = []
    def notify(self, algorithm):
        self.data["F"].append(algorithm.pop.get("F"))
        self.data["X"].append(algorithm.pop.get("X"))
        
class MyCallbackHV(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["F"] = []
        self.data["X"] = []
        self.data["HV"] = []
        self.data["G"] = []
        self.data["COND"] = []
        self.data["LOSS"] = []
        # self.hv = get_performance_indicator("hv", ref_point=np.array([1., 1.]))
    def notify(self, algorithm):
        self.data["F"].append(algorithm.pop.get("F"))
        self.data["X"].append(algorithm.pop.get("X"))
        # self.data["HV"].append(self.hv.calc(algorithm.pop.get("F")))
        self.data["G"].append(algorithm.pop.get("G"))
        self.data["COND"].append(algorithm.pop.get("COND"))
        self.data["LOSS"].append(algorithm.pop.get("LOSS"))