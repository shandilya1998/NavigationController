import numpy as np
import torch
import torchvision as tv
from constants import params
import stable_baselines3 as sb3
import gym

def check_for_nan(inp, name):
    if torch.isnan(inp).any():
        print('nan in {}'.format(name))

class BasalGanglia(torch.nn.Module):
    def __init__(self,
        num_out = 2, 
        num_ctx = 300,
        num_gpe = 40, 
        num_stn = 40,
        num_gpi = 20,
        FF_Dim_in = 20, 
        FF_steps = 2, 
        stn_gpe_iter = 2, 
        eta_gpe = 0.01,
        eta_gpi = 0.01,
        eta_th = 0.01,
    ):  
        super(BasalGanglia, self).__init__()
        self.num_out = num_out
        self.num_ctx = num_ctx
        self.num_gpe = num_gpe
        self.num_stn = num_stn
        self.num_gpi = num_gpi
        self.FF_Dim_in = FF_Dim_in
        self.FF_steps = FF_steps
        self.stn_gpe_iter = stn_gpe_iter
        self.eta_gpe = eta_gpe
        self.eta_gpi = eta_gpi
        self.eta_stn = eta_gpe / 3 
        self.eta_th = eta_th

        input_size = num_ctx
        layers = []
        for units in params['snc']:
            layers.append(torch.nn.Linear(input_size, units))
            if units != 1:
                layers.append(torch.nn.ReLU())
            input_size = units

        self.vf = torch.nn.Sequential(
            *layers
        )

        self.log_a1 = torch.nn.Parameter(torch.Tensor(np.array([[1.0]])))
        self.log_a2 = torch.nn.Parameter(torch.Tensor(np.array([[1.0]])))
        self.thetad1 = torch.nn.Parameter(torch.Tensor(np.array([[0.0]])))
        self.thetad2 = torch.nn.Parameter(torch.Tensor(np.array([[0.0]])))
        self.wsg = torch.nn.Parameter(torch.Tensor(np.array([[2.0]])))
        self.wgs = torch.nn.Parameter(torch.Tensor(np.array([[-2.0]])))
        self.epsilon_glat = torch.nn.Parameter(torch.Tensor(np.array([[.05]])))
        self.weights_glat = torch.ones(
            (self.num_gpe, self.num_gpe)
        ) - torch.eye(self.num_gpe)
        self.epsilon_slat = torch.nn.Parameter(torch.Tensor(np.array([[.05]])))
        self.weights_slat = torch.ones(
            (self.num_stn, self.num_stn)
        ) - torch.eye(self.num_stn) 
        self.fc_d1gpi = torch.nn.Linear(self.FF_Dim_in, self.num_gpi, bias = False)
        self.fc_stngpi = torch.nn.Linear(self.num_stn, self.num_gpi, bias = False)
        self.fc_stngpi.weight.requires_grad = False
        self.fc_jd1 = torch.nn.Linear(self.num_ctx, self.FF_Dim_in, bias = False)
        self.fc_jd2 = torch.nn.Linear(self.num_ctx, self.FF_Dim_in, bias = False)
        self.fc_kd1 = torch.nn.Linear(self.num_ctx, self.FF_Dim_in, bias = False)
        self.fc_kd2 = torch.nn.Linear(self.num_ctx, self.FF_Dim_in, bias = False)

        self.thalamus = torch.nn.RNNCell(
            self.num_gpi, self.num_gpi
        )
        self.linear = torch.nn.Linear(self.num_gpi, self.num_out)

    def forward(self, inputs):
        stimulus_t, stimulus_t_1 = inputs
        batch_size = stimulus_t.shape[0]
        v_t = self.vf(stimulus_t)
        v_t_1 = self.vf(stimulus_t_1)
        deltavf = v_t - v_t_1
        V_D1 = torch.zeros((batch_size, self.FF_Dim_in)).to(stimulus_t.device)
        V_D2 = torch.zeros((batch_size, self.FF_Dim_in)).to(stimulus_t.device)
        lamd1 = 1 / (1 + torch.exp(-self.log_a1.exp() * (deltavf - self.thetad1)))
        lamd2 = 1 / (1 + torch.exp(self.log_a2.exp() * (deltavf - self.thetad2)))

        J_D1 = self.fc_jd1(stimulus_t)
        J_D2 = self.fc_jd2(stimulus_t)
        K_D1 = self.fc_kd1(stimulus_t)
        K_D2 = self.fc_kd2(stimulus_t)

        for FFiter in range(self.FF_steps):
            V_D1 = J_D1 * (1 - V_D1) + (1 - K_D1) * V_D1
            V_D2 = J_D2 * (1 - V_D2) + (1 - K_D2) * V_D2
            V_D1 = torch.sigmoid(lamd1 * V_D1)
            V_D2 = torch.sigmoid(lamd2 * V_D2)
        V_GPi_DP = self.fc_d1gpi(V_D1)
        V_GPi = torch.zeros((batch_size, self.num_gpi)).to(stimulus_t.device)
        xgpe = torch.zeros((batch_size, self.num_gpe)).to(stimulus_t.device)
        xstn = torch.zeros((batch_size, self.num_stn)).to(stimulus_t.device)
        vstn = torch.tanh(lamd2 * xstn)
        hx = torch.rand((batch_size, self.num_gpi)).to(stimulus_t.device)
        for it in range(self.stn_gpe_iter):
            dxgpe = self.eta_gpe * (
                -xgpe + self.wsg * vstn + \
                    torch.nn.functional.linear(
                        xgpe,
                        self.epsilon_glat * self.weights_glat.to(stimulus_t.device) + \
                            torch.ones((
                                self.num_gpe, self.num_gpe
                            )).to(stimulus_t.device)
                    ) - V_D2
                )
            xgpe = xgpe + dxgpe
            dxstn = self.eta_stn * (
                -xstn + self.wgs * xgpe + \
                    torch.nn.functional.linear(
                        vstn,
                        self.epsilon_slat * self.weights_slat.to(stimulus_t.device)
                    )
            )
            xstn = xstn + dxstn
            vstn = torch.tanh(lamd2 * xstn)
            V_GPi_IP = lamd2 * self.fc_stngpi(vstn)
            dvgpi = self.eta_gpi * (-V_GPi - V_GPi_DP + 2 * V_GPi_IP)
            V_GPi = V_GPi + dvgpi
            Ith = -V_GPi
            hx = self.thalamus(Ith, hx)
            out = self.linear(hx)
        return out, v_t

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class VisualCortex(torch.nn.Module):
    def __init__(self, 
        observation_space,
        num_ctx = 300,
    ):
        super(VisualCortex, self).__init__()
        img_obs = observation_space['observation']
        if len(img_obs.shape) == 4:
            img_obs = gym.spaces.Box(
                low = img_obs.low[0],
                high = img_obs.high[0] * 255.0,
                dtype = np.uint8,
                shape = img_obs.shape[1:]
            )
        self.model = sb3.common.torch_layers.NatureCNN(img_obs, num_ctx)
    
    def forward(self, img):
        return self.model(img)

class MotorCortex(torch.nn.Module):
    def __init__(self, num_ctx = 300, action_dim = 2):
        super(MotorCortex, self).__init__()
        layers = []
        input_size = num_ctx
        for units in params['motor_cortex']:
            layers.append(torch.nn.Linear(input_size, units))
            layers.append(torch.nn.ReLU())
            input_size = units
        layers.append(torch.nn.Linear(input_size, action_dim))
        self.squash_fn = torch.nn.Tanh()
        self.fc_2 = torch.nn.Sequential(
            *layers
        )

    def forward(self, inputs):
        stimulus, bg_out = inputs
        action = self.squash_fn(self.fc_2(stimulus) + bg_out)
        return action

class ControlNetwork(torch.nn.Module):
    def __init__(self,
        action_dim = 2,
        num_gpe = 40, 
        num_stn = 40,
        num_gpi = 20,
        FF_Dim_in = 40, 
        FF_steps = 20, 
        stn_gpe_iter = 50, 
        eta_gpe = 0.01,
        eta_gpi = 0.01,
        eta_th = 0.01,
    ):
        super(ControlNetwork, self).__init__()
        num_ctx = params['num_ctx']
        self.bg = BasalGanglia(
            action_dim,
            num_ctx,
            num_gpe, 
            num_stn,
            num_gpi,
            FF_Dim_in, 
            FF_steps, 
            stn_gpe_iter, 
            eta_gpe,
            eta_gpi,
            eta_th,
        )
        self.mc = MotorCortex(
            num_ctx, action_dim
        )

    def forward(self, inputs): 
        stimulus_t, stimulus_t_1 = inputs
        bg_out, vt  = self.bg([stimulus_t, stimulus_t_1])
        action = self.mc([stimulus_t, bg_out])
        return action, vt, bg_out

class MotorCortexV2(torch.nn.Module):
    def __init__(self, num_ctx = 300, action_dim = 2): 
        super(MotorCortexV2, self).__init__()
        layers = []
        input_size = num_ctx
        for units in params['motor_cortex']:
            layers.append(torch.nn.Linear(input_size, units))
            layers.append(torch.nn.ReLU())
            input_size = units
        layers.append(torch.nn.Linear(input_size, action_dim))
        self.squash_fn = torch.nn.Tanh()
        self.fc_2 = torch.nn.Sequential(
            *layers
        )   

    def forward(self, inputs):
        stimulus = inputs
        action = self.squash_fn(self.fc_2(stimulus))
        return action

class ControlNetworkV2(torch.nn.Module):
    def __init__(self,
        action_dim = 2,
    ):  
        super(ControlNetworkV2, self).__init__()
        num_ctx = params['num_ctx']
        input_size = num_ctx
        layers = []
        for units in params['snc']:
            layers.append(torch.nn.Linear(input_size, units))
            if units != 1:
                layers.append(torch.nn.ReLU())
            input_size = units

        self.vf = torch.nn.Sequential(
            *layers
        )
        self.mc = MotorCortexV2(
            num_ctx, action_dim
        )   

    def forward(self, inputs): 
        stimulus_t = inputs
        vt  = self.vf(stimulus_t)
        action = self.mc(stimulus_t)
        bg_out = torch.zeros_like(action).to(action.device)
        return action, vt, bg_out
