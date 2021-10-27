import numpy as np
import torch

class BasalGanglia(torch.nn.Module):
    def __init__(self,
        timesteps,
        num_out = 2,
        num_ctx = 3000,
        num_str = 20,
        num_D1 = 20,
        num_D2 = 20,
        num_gpe = 20,
        num_stn = 3000,
        num_gpi = 2,
        num_snc = 1,
        FF_Dim_in = 20,
        FF_steps = 20,
        stn_gpe_iter = 50,
        eta_gpe = 1,
        eta_gpi = 0.1,
        eta_th - 0.01,
        wsg = 2,
        wgs = -2,
        a1 = 1,
        a2 = 1,
        thetad1 = 0,
        thetad2 = 0,
    ):
        self.timesteps = timesteps
        self.num_out = num_out
        self.num_ctx = num_ctx
        self.num_str = num_str
        self.num_D1 = num_D1
        self.num_D2 = num_D2
        self.num_gpe = num_gpe
        self.num_stn = num_stn
        self.num_gpi = num_gpi
        self.num_snc = num_snc
        self.FF_Dim_in = FF_Dim_in
        self.FF_steps = FF_steps,
        self.stn_gpe_iter = stn_gpe_iter
        self.eta_gpe = eta_gpe
        self.eta_stn = eta_gpe / 3
        self.eta_th = eta_th
        self.wsg = wsg
        self.wgs = wgs
        self.a1 = a1
        self.a2 = a2
        self.thetad1 = thetad1
        self.thetad2 = thetad2

        self.fc_sg = torch.nn.Linear(self.num_stn, self.num_gpe)
        self.fc_gs = torch.nn.Linear(self.num_gpe, self.num_stn)
        self.fc_glat = torch.nn.Linear(self.num_gpe, self.num_gpe)
        self.fc_slat = torch.nn.Linear(self.num_stn, self.num_stn)
        self.fc_d1gpi = torch.nn.Linear(self.num_d1, self.num_gpi)
        self.fc_stngpi = torch.nn.Linear(self.num_stn, self.num_gpi)
        self.W_J_D1 = torch.nn.Parameter(
            torch.rand((num_ctx, FF_Dim_in)) * 0.01)
        self.W_K_D1 = torch.nn.Parameter(
            torch.rand((num_ctx, FF_Dim_in)) * 0.01)
        self.W_J_D2 = torch.nn.Parameter(
            torch.rand((num_ctx, FF_Dim_in)) * 0.01)
        self.W_K_D2 = torch.nn.Parameter(
            torch.rand((num_ctx, FF_Dim_in)) * 0.01)

        self.thalamus = torch.nn.LSTMCell(
            self.num_gpi, self.num_out
        )

    def forward(self, stimulus, deltavf):
        batch_size = stimulus.shape[0]
        V_D1 = torch.zeros((1,FF_Dim_in))
        V_D2 = torch.zeros((1,FF_Dim_in))
        
        lamd1 = 1 / (1 + torch.exp(-self.a1 * (deltavf - self.thetad1)))
        lamd2 = 1 / (1 + torch.exp(-self.a2 * (deltavf - self.thetad2)))

        for FFiter in range(FFsteps):
            J_D1 = stimulus * self.W_J_D1
            K_D1 = stimulus * self.W_K_D1
            K_D2 = stimulus * self.W_K_D2
            V_D1= J_D1 * (1 - V_D1) + (1 - K_D1) * V_D1
            V_D2= J_D2 * (1 - V_D2) + (1 - K_D2) * V_D2

        V_GPi_DP = self.fc_d1gpi(V_D1) 
        V_GPi = torch.zeros((batch_size, self.num_gpi))
        xgpe = torch.zeros((batch_size, self.num_gpe))
        xstn = torch.zeros((batch_size, self.num_stn))
        vstn = torch.nn.functional.tanh(lamd2 * xstn)
        xth = torch.rand((batch_size, self.num_out))
        for it in range(self.stn_gpe_iter):
            dxgpe = self.eta_gpe * (
                -xgpe + self.fc_sg(vstn) + self.fc_glat(xgpe) - V_D2)
            xgpe = xgpe + dxgpe
            dxstn = self.eta_stn * (
                -xstn + self.fc_gs(xgpe) + self.fc_slat(vstn))
            xstn = xstn + dxstn
            vstn = torch.nn.functional.tanh(lamd2 * xstn)
            V_GPi_IP = lamd2 * self.fc_stngpi(vstn)
            dvgpi = self.eta_gpi * (-V_GPi - V_GPi_DP + 2 * V_GPi_IP)
            V_GPi = V_GPi + dvgpi
            Ith = -V_GPi
            xth = self.thalamus(Ith, xth)
        return xth
