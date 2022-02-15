import numpy as np
import pandas as pd
from constants import params
import matplotlib.pyplot as plt
from tqdm import tqdm

def findLocalMaximaMinima(n, arr):

    # Empty lists to store points of
    # local maxima and minima
    mx = []
    mn = []

    # Checking whether the first point is
    # local maxima or minima or neither
    if(arr[0] > arr[1]):
        mx.append(0)
    elif(arr[0] < arr[1]):
        mn.append(0)

    # Iterating over all points to check
    # local maxima and local minima
    for i in range(1, n-1):

        # Condition for local minima
        if(arr[i-1] > arr[i] < arr[i + 1]):
            mn.append(i)

        # Condition for local maxima
        elif(arr[i-1] < arr[i] > arr[i + 1]):
            mx.append(i)

    # Checking whether the last point is
    # local maxima or minima or neither
    if(arr[-1] > arr[-2]):
        mx.append(n-1)
    elif(arr[-1] < arr[-2]):
        mn.append(n-1)

        # Print all the local maxima and
        # local minima indexes stored
    return np.array(mx), np.array(mn)

def test_cpg_entrainment(cpg, C, hopf_mod, hopf):
    omega = np.random.uniform(low = 0.0, high = 2 * np.pi, size = (1,))
    mu = np.random.uniform(low = 0.0, high = 1.0, size = (1,))
    phase = np.random.uniform(low = 0.0, high = 2 * np.pi, size = (1,))
    N = 100000
    dt = 0.005
    lambdas = np.power(10.0, np.arange(-2, 2, 1)).tolist()
    fig, ax = plt.subplots(3, 3, figsize = (22.5, 22.5))
    steps = int(2 * np.pi / (2 * omega[0] * dt * params['alpha']))
    z = mu * np.concatenate([np.cos(phase), np.sin(phase)], -1)
    Z, W = hopf_mod(omega, mu, z, C, params['degree'], N, dt)
    z = mu * np.array([1, 0])
    x, y = np.split(Z, 2, -1)
    phi_ref = np.arctan2(y, x)[:-(Z.shape[0] % steps)]
    def func(x):
        if x < 0:
            return x + 2 * np.pi
        else:
            return x
    func = np.vectorize(func)
    phi_ref = func(phi_ref)
    r_ref = np.sqrt(np.square(x) + np.square(y))
    amp = np.max(r_ref)
    Z_d = hopf(omega, mu, z, N, dt)
    x_d, y_d = np.split(Z_d, 2, -1) 
    phi_d = np.arctan2(y_d, x_d)[:-(Z_d.shape[0] % steps)]
    phi_d = func(phi_d)
    Z_d = amp * np.concatenate([
        np.cos(phi_d),
        np.sin(phi_d)
    ], -1)
    T = np.arange(N) * dt
    ax[1][0].plot(
        T[-steps:], Z[-steps:, 0],
        color = 'r',
        label = 'reference'
    )
    ax[1][1].plot(
        T[-steps:], Z[-steps::, 1],
        color = 'r',
        label = 'reference'
    )
    ax[2][0].plot(T[-steps:], phi_ref[-steps:],
        color = 'r',
        label = 'reference'
    )

    ax[1][0].plot(
        T[-steps:], Z_d[-steps:, 0], 
        color = 'b',
        label = 'driver'
    )   
    ax[1][1].plot(
        T[-steps:], Z_d[-steps::, 1], 
        color = 'b',
        label = 'driver'
    )   
    ax[2][0].plot(T[-steps:], phi_d[-steps:],
        color = 'b',
        label = 'driver'
    )

    for lmbda in tqdm(lambdas):
        params['lambda'] = lmbda
        Z2, W2, Z1 = cpg(omega, mu, phase, C, params['degree'], N, dt)
        length = int(Z2.shape[0] - Z2.shape[0] % steps)
        x, y = np.split(Z2, 2, -1)
        r = np.linalg.norm(
            np.sqrt(np.square(x) + np.square(y)) - r_ref,
            axis = -1
        )
        phi = func(np.arctan2(y, x))[:length, :]
        err = np.sqrt(np.square(
            phi - phi_ref,
        ))
        err = err.reshape(int(err.shape[0] / steps), steps, err.shape[-1])
        err = np.sum(err, 1) / steps
        err2 = np.sqrt(np.square(W2  - W))[:length, :]
        err2 = err2.reshape(int(err2.shape[0] / steps), steps, err2.shape[-1])
        err2 = np.sum(err2, 1) / steps
        err3 = np.sqrt(np.square(
            np.gradient(phi, axis = 0) - np.gradient(phi_ref, axis = 0)
        ))
        err3 = err3.reshape(int(err3.shape[0] / steps), steps, err3.shape[-1])
        err3 = np.sum(err3, 1) / steps
        err4 = np.sqrt(np.square(
            phi - phi_d[:length, :]
        ))
        err4 = err4.reshape(int(err4.shape[0] / steps), steps, err4.shape[-1])
        err4 = np.sum(err4, 1) / steps
        color = np.random.uniform(low = 0.0, high = 1.0, size = (3,))
        ax[0][0].plot(T, r,
            color = color,
            label = '\u03BB: {}'.format(lmbda),
            linestyle = '--')
        ax[0][1].plot(np.arange(err.shape[0]), err,
            color = color,
            label = '\u03BB: {}'.format(lmbda),
            linestyle = '--')
        ax[0][2].plot(T, r,
            color = color,
            label = '\u03BB: {}'.format(lmbda),
            linestyle = '--')
        ax[1][0].plot(
            T[-steps:], Z2[-steps:, 0],
            color = color,
            label = '\u03BB: {}'.format(lmbda),
            linestyle = '--')
        ax[1][1].plot(
            T[-steps:], Z2[-steps:, 1],
            color = color,
            label = ' \u03BB: {}'.format(lmbda),
            linestyle = '--')
        ax[1][2].plot(np.arange(err3.shape[0]), err3,
            color = color,
            label = '\u03BB: {}'.format(lmbda),
            linestyle = '--')
        ax[2][0].plot(T[-steps:], phi[-steps:],
            color = color,
            label = '\u03BB: {}'.format(lmbda),
            linestyle = '--')
        ax[2][1].plot(np.arange(err2.shape[0]), err2,
            color = color,
            label = '\u03BB: {}'.format(lmbda),
            linestyle = '--')
        ax[2][2].plot(np.arange(err4.shape[0]), err4,
            color = color,
            label = '\u03BB: {}'.format(lmbda),
            linestyle = '--'
        )    
    ax[0][0].set_xlabel('time')
    ax[0][0].set_ylabel('err in r')
    ax[0][1].set_xlabel('no. of periods')
    ax[0][1].set_ylabel('err in phase wrt reference')
    ax[0][2].set_xlabel('time')
    ax[0][2].set_ylabel('r')
    ax[1][0].set_xlabel('time')
    ax[1][0].set_ylabel('real part')
    ax[1][1].set_xlabel('time')
    ax[1][1].set_ylabel('imaginary part')
    ax[1][2].set_xlabel('no of periods')
    ax[1][2].set_ylabel('err in gradient of phase')
    ax[2][0].set_xlabel('time')
    ax[2][0].set_ylabel('phase')
    ax[2][1].set_xlabel('no. of periods')
    ax[2][1].set_ylabel('err in phase velocity')
    ax[2][2].set_xlabel('no of periods')
    ax[2][2].set_ylabel('err in phase wrt driver')
    ax[0][0].legend(loc = 'upper right')
    ax[0][1].legend(loc = 'upper right')
    ax[0][2].legend(loc = 'upper right')
    ax[1][0].legend(loc = 'upper right')
    ax[1][1].legend(loc = 'upper right')
    ax[1][2].legend(loc = 'upper right')
    ax[2][0].legend(loc = 'upper right')
    ax[2][1].legend(loc = 'upper right')
    ax[2][2].legend(loc = 'upper right')
    fig.savefig('test.png')
    plt.show()
