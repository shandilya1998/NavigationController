import numpy as np
import torch
import torchvision as tv
from constants import params


class BasalGanglia(torch.nn.Module):
    def __init__(self,
        num_out = 20, 
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
        wsg = 2,
        wgs = -2, 
        a1 = 1,
        a2 = 1,
        thetad1 = 0,
        thetad2 = 0,
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
        self.a1 = a1
        self.a2 = a2
        self.thetad1 = thetad1
        self.thetad2 = thetad2

        self.wgs = torch.nn.Parameter(torch.Tensor(np.array([[2.0]])))
        self.epsilon_glat = torch.nn.Parameter(torch.Tensor(np.array([[.05]])))
        self.weights_glat = torch.ones(
            (self.num_gpe, self.num_gpe)
        ) - torch.eye(self.num_gpe)
        self.epsilon_slat = torch.nn.Parameter(torch.Tensor(np.array([[.05]])))
        self.weights_slat = torch.ones(
            (self.num_stn, self.num_stn)
        ) - torch.eye(self.num_stn) 
        self.fc_d1gpi = torch.nn.Linear(self.FF_Dim_in, self.num_gpi)
        self.fc_stngpi = torch.nn.Linear(self.num_stn, self.num_gpi)
        self.fc_jd1 = torch.nn.Linear(self.num_ctx, self.FF_Dim_in)
        self.fc_jd2 = torch.nn.Linear(self.num_ctx, self.FF_Dim_in)
        self.fc_kd1 = torch.nn.Linear(self.num_ctx, self.FF_Dim_in)
        self.fc_kd2 = torch.nn.Linear(self.num_ctx, self.FF_Dim_in)

        self.thalamus = torch.nn.LSTMCell(
            self.num_gpi, self.num_out
        )

    def forward(self, inputs):
        stimulus, deltavf = inputs
        batch_size = stimulus.shape[0]
        V_D1 = torch.zeros((batch_size, self.FF_Dim_in)).to(stimulus.device)
        V_D2 = torch.zeros((batch_size, self.FF_Dim_in)).to(stimulus.device)
        lamd1 = 1 / (1 + torch.exp(-self.a1 * (deltavf - self.thetad1)))
        lamd2 = 1 / (1 + torch.exp(-self.a2 * (deltavf - self.thetad2)))
        for FFiter in range(self.FF_steps):
            J_D1 = self.fc_jd1(stimulus)
            J_D2 = self.fc_jd2(stimulus)
            K_D1 = self.fc_kd1(stimulus)
            K_D2 = self.fc_kd2(stimulus)
            V_D1= J_D1 * (1 - V_D1) + (1 - K_D1) * V_D1
            V_D2= J_D2 * (1 - V_D2) + (1 - K_D2) * V_D2
            V_D1 = torch.sigmoid(lamd1 * V_D1)
            V_D2 = torch.sigmoid(lamd2 * V_D2)
        V_GPi_DP = self.fc_d1gpi(V_D1)
        V_GPi = torch.zeros((batch_size, self.num_gpi)).to(stimulus.device)
        xgpe = torch.zeros((batch_size, self.num_gpe)).to(stimulus.device)
        xstn = torch.zeros((batch_size, self.num_stn)).to(stimulus.device)
        vstn = torch.tanh(lamd2 * xstn)
        hx = torch.rand((batch_size, self.num_out)).to(stimulus.device)
        cx = torch.rand((batch_size, self.num_out)).to(stimulus.device)
        for it in range(self.stn_gpe_iter):
            dxgpe = self.eta_gpe * (
                -xgpe + self.wgs * vstn + \
                    torch.nn.functional.linear(
                        xgpe,
                        self.epsilon_glat * self.weights_glat.to(stimulus.device) + \
                            torch.ones((
                                self.num_gpe, self.num_gpe
                            )).to(stimulus.device)
                    ) - V_D2
                )
            xgpe = xgpe + dxgpe
            dxstn = self.eta_stn * (
                -xstn - self.wgs * xgpe + \
                    torch.nn.functional.linear(
                        vstn,
                        self.epsilon_slat * self.weights_slat.to(stimulus.device)
                    )
            )
            xstn = xstn + dxstn
            vstn = torch.tanh(lamd2 * xstn)
            V_GPi_IP = lamd2 * self.fc_stngpi(vstn)
            dvgpi = self.eta_gpi * (-V_GPi - V_GPi_DP + 2 * V_GPi_IP)
            V_GPi = V_GPi + dvgpi
            Ith = -V_GPi
            hx, cx = self.thalamus(Ith, (hx, cx))
        return hx

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class VisualCortex(torch.nn.Module):
    def __init__(self, 
        use_pretrained = True,
        feature_extracting = False,
        num_ctx = 300,
    ):
        super(VisualCortex, self).__init__()
        self.model_ft = tv.models.mobilenet_v3_small(pretrained=use_pretrained)
        set_parameter_requires_grad(self.model_ft, feature_extracting)
        num_ftrs = self.model_ft.classifier._modules['3'].in_features
        self.model_ft.classifier._modules['3'] = torch.nn.Linear(num_ftrs, num_ctx)

    def forward(self, img):
        """
            Input to this model need to be preprocessed as follows
            img = torch.unsqueeze(
                torch.permute(
                    torch.from_numpy(
                        ob.astype('float32')/255
                    ), (2, 0, 1)
                ), 0
            )
            where ob is the visual observation from the environment
        """
        return self.model_ft(img)


class MotorCortex(torch.nn.Module):
    def __init__(self, num_ctx = 300, num_bg_out = 20, action_dim = 2):
        super(MotorCortex, self).__init__()
        layers = []
        input_size = num_ctx
        for units in params['motor_cortex'][0]:
            layers.append(torch.nn.Linear(input_size, units))
            layers.append(torch.nn.ReLU())
            input_size = units
        layers.append(torch.nn.Linear(input_size, num_bg_out))
        input_size = num_bg_out
        self.fc_1 = torch.nn.Sequential(
            *layers
        )
        layers = []
        for units in params['motor_cortex'][1]:
            layers.append(torch.nn.Linear(input_size, units))
            layers.append(torch.nn.ReLU())
            input_size = units
        layers.append(torch.nn.Linear(input_size, action_dim))
        layers.append(torch.nn.Tanh())
        self.fc_2 = torch.nn.Sequential(
            *layers
        )

    def forward(self, inputs):
        stimulus, bg_out = inputs
        x = self.fc_1(stimulus)
        x = x + bg_out
        action = self.fc_2(x)
        return action


class ControlNetwork(torch.nn.Module):
    def __init__(self,
        action_dim = 2,
        num_bg_out = 20, 
        num_ctx = 300,
        num_gpe = 40, 
        num_stn = 40,
        num_gpi = 20,
        FF_Dim_in = 40, 
        FF_steps = 20, 
        stn_gpe_iter = 50, 
        eta_gpe = 0.01,
        eta_gpi = 0.01,
        eta_th = 0.01,
        wsg = 2,
        wgs = -2, 
        a1 = 1,
        a2 = 1,
        thetad1 = 0,
        thetad2 = 0,
        use_pretrained_visual_cortex = True,
        feature_extracting_visual_cortex = False
    ):
        super(ControlNetwork, self).__init__()
        self.bg = BasalGanglia(
            num_bg_out,
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
            wsg,
            wgs, 
            a1,
            a2,
            thetad1,
            thetad2,
        )
        self.vc = VisualCortex(
            use_pretrained_visual_cortex,
            feature_extracting_visual_cortex,
            num_ctx
        )
        self.mc = MotorCortex(
            num_ctx, num_bg_out, action_dim
        )

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
        
        input_size = num_ctx + action_dim
        layers = []
        for units in params['snc']:
            layers.append(torch.nn.Linear(input_size, units))
            if units != 1:
                layers.append(torch.nn.ReLU())
            input_size = units

        self.af = torch.nn.Sequential(
            *layers
        )

    def forward(self, inputs): 
        stimulus, vt_1 = inputs
        stimulus = self.vc(stimulus)
        vt = self.vf(stimulus)
        deltavf = vt - vt_1
        bg_out  = self.bg([stimulus, deltavf])
        action = self.mc([stimulus, bg_out])
        at = self.af(torch.cat([stimulus, action], -1))
        return action, vt, at
