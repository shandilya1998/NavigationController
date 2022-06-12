from neurorobotics.reward.support_plane_v2 import SupportPlane
import numpy as np

class ZMP:
    def __init__(self, params):
        self.params = params
        self.support_plane = SupportPlane(params)
        self.g = np.array([
            0.0,
            0.0,
            -9.8
        ])
        self.zmp_s = np.zeros((3,))
        self.zmp = np.zeros((3,))
        self.inertial_plane = np.eye(N = 3)
        self.plane = self.inertial_plane

    def update_g(self, g):
        self.g = g

    def build(self, t, Tb, A, B, AL, BL, AF, BF):
        self.support_plane.build(t, Tb, A, B, AL, BL, AF, BF)

    def _transform(self, vec, transform):
        return self.support_plane.transform(vec, transform)

    def get_ZMP_s(self, com, acc):
        self.plane = self.support_plane()
        transform = self.support_plane.transformation_matrix(
            self.plane,
            self.inertial_plane
        )
        com_s = self._transform(com, transform)
        acc_s = self._transform(acc, transform)
        g_s = self._transform(self.g, transform)
        zmp_s = np.zeros((3,))
        zmp_s[1] = com_s[1] - (
            com_s[0] * (
                acc_s[1] + g_s[1]
            )
        ) / (acc_s[0] + g_s[0])
        zmp_s[2] = com_s[2] - (
            com_s[0] * (
                acc_s[2] + g_s[2]
            )
        ) / (acc_s[0] + g_s[0])
        return zmp_s

    def __call__(self, com, acc, v_real, v_exp, eta):
        self.zmp_s = self.get_ZMP_s(com, acc)
        transform = self.support_plane.transformation_matrix(
            self.plane,
            self.inertial_plane
        )
        v_real = self._transform(
            v_real,
            transform
        )
        v_exp = self._transform(
            v_exp,
            transform
        )
        self.zmp = self.zmp_s + eta*(v_real - v_exp)
        self.zmp[0] = 0
        transform = self.support_plane.transformation_matrix(
            self.inertial_plane,
            self.plane
        )
        return self._transform(self.zmp, transform)
