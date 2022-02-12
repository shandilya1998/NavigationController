import numpy as np

class SupportPlane:
    def __init__(self, params):
        self.params = params

    def build(self, t, Tb, A, B, AL, BL, AF, BF):
        self.t = t
        self.flag = False
        if Tb == 0:
            Tb = 1e-8
        self.Tb = Tb
        self.A = A
        self.AL = AL
        self.AF = AF
        self.B = B
        self.BL = BL
        self.BF = BF
        self.AB = self.B - self.A
        self.AAf = self.AF - self.A
        self.BBf = self.BF - self.B
        self.AAl = self.AL - self.A
        self.BBl = self.BL - self.B

    def get_n11(self):
        cross = np.cross(self.AB, self.AAf)
        norm = np.linalg.norm(cross)
        if norm == 0:
            cross = np.cross(self.AB, self.BBf)
            norm = np.linalg.norm(cross)
        if norm == 0:
            norm = 1e-8
            self.flag = True
        n11 = cross / norm
        if n11[0] < 0:
            return -1 * n11
        else:
            return n11

    def get_n12(self):
        cross = np.cross(self.AB, self.BBf)
        norm = np.linalg.norm(cross)
        if norm == 0:
            cross = np.cross(self.AB, self.AAf)
            norm = np.linalg.norm(cross)
        if norm == 0:
            norm = 1e-8
            self.flag = True
        n12 = cross / norm
        if n12[0] < 0:
            return -1 * n12
        else:
            return n12

    def get_n21(self):
        cross = np.cross(self.AB, self.AAl)
        norm = np.linalg.norm(cross)
        if norm == 0:
            cross = np.cross(self.AB, self.BBl)
            norm = np.linalg.norm(cross)
        if norm == 0:
            norm = 1e-8
            self.flag = True
        n21 = cross / norm
        if n21[0] < 0:
            return -1 * n21
        else:
            return n21

    def get_n22(self):
        cross = np.cross(self.AB, self.BBl)
        norm = np.linalg.norm(cross)
        if norm == 0:
            cross = np.cross(self.AB, self.AAl)
            norm = np.linalg.norm(cross)
        if norm == 0:
            norm = 1e-8
            self.flag = True
        n22 = cross / norm
        if n22[0] < 0:
            return -1 * n22
        else:
            return n22

    def get_n1(self):
        n11 = self.get_n11()
        n12 = self.get_n12()
        norm = np.linalg.norm(n11 + n12)
        if norm == 0:
            self.flag = True
            return n11 + n12
        return (n11 + n12)/np.linalg.norm(n11 + n12)

    def get_n2(self):
        n21 = self.get_n21()
        n22 = self.get_n22()
        norm = np.linalg.norm(n21 + n22)
        if norm == 0:
            self.flag = True
            return n21 + n22
        return (n21 + n22)/np.linalg.norm(n21 + n22)

    def get_xs(self, t):
        mu = -t/self.Tb + 1
        n1 = self.get_n1()
        n2 = self.get_n2()
        temp = mu*n1 + (1-mu)*n2
        norm = np.linalg.norm(temp)
        if norm == 0.0:
            norm = 1.0
        return temp/norm

    def get_zs(self):
        norm = np.linalg.norm(self.AB)
        if norm == 0.0:
            norm = 1.0
        return self.AB/norm

    def get_ys(self, t, xs, zs):
        return np.cross(zs, xs)

    def transformation_matrix(self, cs1, cs2):
        transform = np.array([
            [
               np.dot(cs1[i], cs2[j]) for j in range(3)
            ] for i in range(3)
        ])
        return transform

    def transform(self, vec, transform):
        """
            Transform a vector vec from cs2 to cs1
        """
        return np.matmul(transform, vec)

    def __call__(self):
        xs = self.get_xs(self.t)
        zs = self.get_zs()
        ys = self.get_ys(self.t, xs, zs)
        plane = np.zeros((3, 3))
        try:
            plane[0, :] = xs/np.linalg.norm(xs)
            plane[1, :] = ys/np.linalg.norm(ys)
            plane[2, :] = zs/np.linalg.norm(zs)
        except Exception as e:
            print('xs {}'.format(xs))
            print('ys {}'.format(ys))
            print('zs {}'.format(zs))
            print('A {}'.format(self.A))
            print('B {}'.format(self.B))
            print('AL {}'.format(self.AL))
            print('BL {}'.format(self.BL))
            print('AF {}'.format(self.AF))
            print('BF {}'.format(self.BF))
            print(e)
        return plane
