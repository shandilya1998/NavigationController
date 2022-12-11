"""Module for CPG based Gait Generation.
"""
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import shutil
from tqdm import tqdm
from neurorobotics.constants import params


def get_pattern(thresholds, dx = 0.001):
    """Returns distribution of beta with respect to omega

    :param thresholds: boundaries of change in beta
    :type thresholds: List[float]
    :param dx: delta change in omega, int(1/dx) gives length of output list
    :type dx: float
    :returns: distribution of beta with respect to ome
    :rtype: List[float]
    """
    out = []
    x = 0.0
    y = [0.9, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25]
    while x < thresholds[1]:
        out.append(((y[1] - y[0])/(thresholds[1] - thresholds[0])) * (x - thresholds[0]) + y[0])
        x += dx
    while x < thresholds[2]:
        out.append(y[2])
        x += dx
    while x < thresholds[3]:
        out.append(((y[3] - y[2])/(thresholds[3] - thresholds[2])) * (x - thresholds[2]) + y[2])
        x += dx
    while x < thresholds[4]:
        out.append(y[4])
        x += dx
    while x < thresholds[5]:
        out.append(((y[5] - y[4])/(thresholds[5] - thresholds[4])) * (x - thresholds[4]) + y[4])
        x += dx
    while x < thresholds[6]:
        out.append(y[6])
        x += dx
    out = np.array(out, dtype = np.float32)
    return out


def get_polynomial_coef(degree, thresholds, dt = 0.001):
    """Returns the coefficients of the polynomial approximating desired beta vs omega dis

    """
    y = get_pattern(thresholds, dt)
    x = np.arange(0, thresholds[-1], dt, dtype = np.float32)
    C = np.polyfit(x, y, degree)
    return C


def plot_beta_polynomial(logdir, C, degree, thresholds, dt = 0.001):
    y = get_pattern(thresholds, dt)
    x = np.arange(0, thresholds[-1], dt, dtype = np.float32)
    def f(x, degree):
        X = np.array([x ** pow for pow in range(degree, -1, -1 )], dtype = np.float32)
        return np.sum(C * X)
    y_pred = np.array([f(x_, degree) for x_ in x], dtype = np.float32)
    fig, ax = plt.subplots(1, 1, figsize = (5,5))
    ax.plot(x, y, color = 'r', linestyle = ':', label = 'desired beta')
    ax.plot(x, y_pred, color = 'b', linestyle = '--', label = 'actual beta')
    ax.set_xlabel('omega')
    ax.set_ylabel('beta')
    ax.legend()
    fig.savefig(os.path.join(logdir, 'polynomial.png'))
    plt.close()
    print('Beta Approximation Plot Done.')


class HopfCPG:
    """NumPy Model for a hopf oscillator.

    :param num_osc: Number of Oscillators in the CPG
    :type num_osc: int
    """
    def __init__(self, num_osc):
        super(HopfCPG, self).__init__()
        self.num_osc = num_osc

    def forward(self, omega, mu, Z, dt=0.001):
        """Feedforward method for Modified Hopf Oscillator.

        :param Z: Current State of the oscillator
        :type Z: torch.Tensor
        """
        assert Z.shape[-1] // 2 == self.num_osc
        x, y = np.split(Z, 2, -1)
        r = np.sqrt(np.square(x) + np.square(y))
        phi = np.arctan2(y, (x + 1e-7))
        w = np.abs(omega) * 2 * params['alpha']
        phi += dt * w
        r += dt * (mu - np.square(r)) * r
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.concatenate([x, y], -1)
        return z


def hopf(num_osc, omega, mu, z, N, dt):
    Z = []
    hopf_simple_step = HopfCPG(num_osc).forward
    for i in range(N):
        z = hopf_simple_step(omega, mu, z, dt)
        Z.append(z.copy())
    return np.stack(Z, 0)


def get_omega_choice(phi):
    return np.tanh(1e3 * (phi))


def get_beta(x, C, degree):
    x = np.abs(x)
    X = np.stack([x ** p for p in range(degree, -1, -1 )], 0)
    return np.array([np.sum(C * X[:, i]) for i in range(X.shape[-1])], dtype = np.float32)


class ModifiedHopfCPG:
    """NumPy Model for a modified hopf oscillator for gait generation.

    :param num_legs: Number of Legs in the robot
    :type num_legs: int
    """
    def __init__(self, num_osc):
        super(ModifiedHopfCPG, self).__init__()
        self.num_osc = num_osc

    def forward(self, omega, mu, Z, C, degree, dt=0.001):
        """Feedforward method for Modified Hopf Oscillator.

        :param Z: Current State of the oscillator
        :type Z: torch.Tensor
        """
        assert Z.shape[-1] // 2 == self.num_osc
        x, y = np.split(Z, 2, -1)
        r = np.sqrt(np.square(x) + np.square(y))
        beta = get_beta(omega, C, degree)
        phi = np.arctan2(y, (x + 1e-7))
        mean = np.abs(1 / (2 * beta * (1 - beta)))
        amplitude = (1 - 2 * beta) / (2 * beta * (1 - beta))
        w = np.abs(omega) * (mean + amplitude * get_omega_choice(phi) * params['alpha'])
        phi += dt * w
        r += params['lambda'] * dt * (mu - params['beta'] * np.square(r)) * r
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.concatenate([x, y], -1)
        return z


def hopf_mod(num_osc, omega, mu, z, C, degree, N, dt):
    Z = []
    hopf_mod_step = ModifiedHopfCPG(num_osc).forward
    for i in range(N):
        z = hopf_mod_step(omega, mu, z, C, degree, dt)
        Z.append(z.copy())
    return np.stack(Z, 0)


def pre_configure_network():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_path',
        type = str,
        help = 'Path to output directory'
    )
    parser.add_argument(
        '--num_osc',
        type = int,
        help = 'number of oscillators'
    )
    parser.add_argument(
        '--timesteps',
        type = int,
        default = 10000,
        help = 'number of timesteps to run oscillators for'
    )
    parser.add_argument(
        '--dt',
        type = float,
        default = 0.001,
        help = 'sampling period'
    )
    args = parser.parse_args()
    num_osc = args.num_osc
    N = args.timesteps
    dt = args.dt
    z = np.concatenate([np.zeros((num_osc,), dtype = np.float32), np.ones((num_osc,), dtype = np.float32)], -1)
    omega = np.arange(1, num_osc + 1, dtype = np.float32) * np.pi * 2 / (num_osc + 1)
    mu = np.ones((num_osc,), dtype = np.float32)
    print('Running Oscillators.')
    plot_path = os.path.join(args.out_path, 'hopf')
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)
    os.mkdir(plot_path)
    C = get_polynomial_coef(params['degree'], params['thresholds'], dt * 50)
    np.save(open(os.path.join(plot_path, 'coef.npy'), 'wb'), C)
    plot_beta_polynomial(plot_path, C, params['degree'], params['thresholds'], dt * 50)
    Z_hopf = hopf(num_osc, omega.copy(), mu.copy(), z.copy(), N, dt)
    Z_mod = hopf_mod(num_osc, omega.copy(), mu.copy(), z.copy(), C, params['degree'], N, dt)
    T = np.arange(N, dtype = np.float32) * dt
    print('Plotting Output.')
    for i in tqdm(range(num_osc)):
        num_steps = int(2 * np.pi / (2 * omega[i] * dt * params['alpha']))
        fig, axes = plt.subplots(2,2, figsize = (12,12))
        axes[0][0].plot(
            T[-num_steps:], Z_hopf[-num_steps:, i],
            linestyle = ':', color = 'r',
            label = 'constant omega'
        )
        axes[0][0].plot(
            T[-num_steps:], Z_mod[-num_steps:, i],
            color = 'b', label = 'variable omega')
        axes[0][0].set_xlabel('time (s)',fontsize=15)
        axes[0][0].set_ylabel('real part',fontsize=15)
        axes[0][0].set_title('Trend in Real Part',fontsize=15)
        axes[0][0].legend()
        axes[0][1].plot(
            T[-num_steps:], Z_hopf[-num_steps:, i + num_osc],
            linestyle = ':', color = 'r',
            label = 'constant omega'
        )
        axes[0][1].plot(T[-num_steps:], Z_mod[-num_steps:, i + num_osc],
            color = 'b', label = 'variable omega'
        )
        axes[0][1].set_xlabel('time (s)',fontsize=15)
        axes[0][1].set_ylabel('imaginary part',fontsize=15)
        axes[0][1].set_title('Trend in Imaginary Part',fontsize=15)
        axes[0][1].legend()
        axes[1][0].plot(
            Z_hopf[:, i], Z_hopf[:, i + num_osc],
            linestyle = ':', color = 'r',
            label = 'constant omega'
        )
        axes[1][0].plot(
            Z_mod[:, i], Z_mod[:, i + num_osc],
            color = 'b', label = 'variable omega'
        )
        axes[1][0].set_xlabel('real part',fontsize=15)
        axes[1][0].set_ylabel('imaginary part',fontsize=15)
        axes[1][0].set_title('Phase Space',fontsize=15)
        axes[1][0].legend()
        axes[1][1].plot(
            T[-num_steps:],
            np.arctan2(
                Z_hopf[-num_steps:, i],
                Z_hopf[-num_steps:, i + num_osc]
            ), linestyle = ':',
            color = 'r', label = 'constant omega'
        )
        axes[1][1].plot(
            T[-num_steps:],
            np.arctan2(
                Z_mod[-num_steps:, i],
                Z_mod[-num_steps:, i + num_osc]
            ), color = 'b',
            label = 'variable omega'
        )
        axes[1][1].set_xlabel('time (s)',fontsize=15)
        axes[1][1].set_ylabel('phase (radians)',fontsize=15)
        axes[1][1].set_title('Trend in Phase',fontsize=15)
        axes[1][1].legend()
        fig.savefig(os.path.join(plot_path, 'oscillator_{}.png'.format(i)))
        plt.close('all')
    phi = np.array([0.0, 0.25, 0.5, 0.75], dtype = np.float32)
    phi = phi + np.cos(phi * 2 * np.pi) * 3 * (1 - 0.75) / 8
    z = np.concatenate([np.cos(phi * 2 * np.pi), np.sin(phi * 2 * np.pi)], -1)
    omega = 1.6 * np.ones((4,), dtype = np.float32)
    mu = np.ones((4,), dtype = np.float32)
    Z_mod = hopf_mod(num_osc, omega.copy(), mu.copy(), z.copy(), C, params['degree'], N, dt)
    fig, axes = plt.subplots(2,2, figsize = (10,10))
    num_osc = 4
    color = ['r', 'b', 'g', 'y']
    for i in tqdm(range(num_osc)):
        num_steps = int(2 * np.pi / (2 * omega[i] * dt * params['alpha']))
        axes[0][0].plot(T[:num_steps], Z_mod[:num_steps, i], color = color[i], linestyle = '--')
        axes[0][0].set_xlabel('time (s)')
        axes[0][0].set_ylabel('real part')
        axes[0][0].set_title('Trend in Real Part')
        axes[0][1].plot(T[:num_steps], -np.maximum(-Z_mod[:num_steps, i + num_osc], 0), color = color[i], linestyle = '--')
        axes[0][1].set_xlabel('time (s)')
        axes[0][1].set_ylabel('imaginary part')
        axes[0][1].set_title('Trend in Imaginary Part')
        axes[1][0].plot(Z_mod[:, i], Z_mod[:, i + num_osc], color = color[i], linestyle = '--')
        axes[1][0].set_xlabel('real part')
        axes[1][0].set_ylabel('imaginary part')
        axes[1][0].set_title('Phase Space')
        axes[1][1].plot(
            T[-num_steps:],
            np.arctan2(Z_mod[-num_steps:, i], Z_mod[-num_steps:, i + num_osc]),
            color = color[i], linestyle = '--'
        )
        axes[1][1].set_xlabel('time (s)')
        axes[1][1].set_ylabel('phase (radian)')
        axes[1][1].set_title('Trend in Phase')
    fig.savefig(os.path.join(plot_path, 'phase_comparison.png'))


class ModifiedHopfCPGTorch(torch.nn.Module):
    """PyTorch Model for a modified hopf oscillator for gait generation.

    :param num_legs: Number of Legs in the robot
    :type num_legs: int
    """
    def __init__(self, num_osc):
        super(ModifiedHopfCPGTorch, self).__init__()
        self.num_osc = num_osc
        self.alpha = params['alpha']
        self.lmbd = params['lambda']
        self.cbeta = params['beta']

    def forward(self, Z, omega, mu, C, degree, dt=0.001):
        """Feedforward method for Modified Hopf Oscillator.

        :param Z: Current State of the oscillator
        :type Z: torch.Tensor
        """
        assert Z.size()[-1] // 2 == self.num_osc
        x, y = torch.split(Z, self.num_osc, -1)
        r = torch.sqrt(torch.square(x) + torch.square(y))
        beta = self._get_beta(omega, C, degree)
        phi = torch.arctan(y / (x + 1e-7))
        mean = torch.abs(1 / (2 * beta * (1 - beta)))
        amplitude = (1 - 2 * beta) / (2 * beta * (1 - beta))
        w = torch.abs(omega) * (mean + amplitude * self._get_omega_choice(phi) * self.alpha)
        phi += dt * w
        r += self.lmbd * dt * (mu - self.beta * torch.square(r)) * r
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = torch.cat([x, y], -1)
        return z

    def _get_omega_choice(self, phi):
        return torch.tanh(1e3 * (phi))

    def _get_beta(self, x, C, degree):
        x = torch.abs(x)
        X = torch.stack([x ** p for p in range(degree, -1, -1)], 1)
        beta = torch.stack([torch.sum(C * X[:, :, i]) for i in range(X.shape[-1])], -1)
        return beta
