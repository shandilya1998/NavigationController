"""Module for CPG based Gait Generation.
"""
import torch


class ModifiedHopfCPG(torch.nn.Module):
    """PyTorch Model for a modified hopf oscillator for gait generation.

    :param num_legs: Number of Legs in the robot
    :type num_legs: int
    """
    def __init__(self, num_legs):
        super(ModifiedHopfCPG, self).__init__()
        self.num_legs = num_legs

    def forward(self, Z, omega, mu, C, degree, alpha, lmbd, cbeta, dt=0.001):
        """Feedforward method for Modified Hopf Oscillator.

        :param Z: Current State of the oscillator
        :type Z: torch.Tensor
        """
        assert Z.size()[-1] // 2 == self.num_legs
        print("Z: ", Z.shape)
        x, y = torch.split(Z, self.num_legs, -1)
        print('x: ', x.shape, ' y: ', y.shape)
        r = torch.sqrt(torch.square(x) + torch.square(y))
        print('r: ', r.shape)
        beta = self._get_beta(omega, C, degree)
        print('beta: ', beta.shape)
        phi = torch.arctan(y / (x + 1e-7))
        print('phi: ', phi.shape)
        mean = torch.abs(1 / (2 * beta * (1 - beta)))
        print('mean: ', mean.shape)
        amplitude = (1 - 2 * beta) / (2 * beta * (1 - beta))
        w = torch.abs(omega) * (mean + amplitude * self._get_omega_choice(phi) * alpha)
        phi += dt * w
        r += lmbd * dt * (mu - cbeta * torch.square(r)) * r
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = torch.cat([x, y], -1)
        return z

    def _get_omega_choice(self, phi):
        return torch.tanh(1e3 * (phi))

    def _get_beta(self, x, C, degree):
        print('x: ', x.shape)
        x = torch.abs(x)
        print('x: ', x.shape)
        X = torch.stack([x ** p for p in range(degree, -1, -1)], 1)
        print('X: ', X.shape)
        print('C: ', C.shape)
        beta = torch.stack([torch.sum(C * X[:, :, i]) for i in range(X.shape[-1])], -1)
        print('beta: ', beta.shape)
        return beta
