from scipy import constants
from numpy import *

def timeStepsNeeded(n, dx, dt):
    return int((n - 1) * dx / constants.c / dt) + 2 # previous timesteps plus current one

def computeR2(n, dx, a, navg):
    R2 = zeros((n - 1, n))
    for i in range(n):
        c = max((i - 0.5) * dx, 0)
        d = min((i + 0.5) * dx, (n - 1) * dx)
        for j in range(n - 1):
            for k in range(navg):
                x = (j + k / (navg - 1.0)) * dx
                if x < c: R2[j, i] += (arctan((c - x) / a) - arctan((d - x) / a))
                elif x > d: R2[j, i] += (arctan((d - x) / a) - arctan((c - x) / a))
                else: R2[j, i] += (-arctan((c - x) / a) - arctan((d - x) / a))
            R2[j, i] /= (4 * pi * constants.epsilon_0 * dx * a * navg)
            if i == 0 or i == n - 1: R2[j, i] *= 2
    return R2

def computeR1L(n, dx, a, navg):
    R1L = zeros((n - 1, n))
    for i in range(n):
        c = max((i - 0.5) * dx, 0)
        d = min((i + 0.5) * dx, (n - 1) * dx)
        for j in range(n - 1):
            for k in range(navg):
                x = (j + k / (navg - 1.0)) * dx
                F = lambda x: log(sqrt(x**2 + a**2) + x)
                if x < c: R1L[j, i] += (F(c - x) - F(d - x))
                elif x > d: R1L[j, i] += (F(d - x) - F(c - x))
                else: R1L[j, i] += (2 * F(0) - F(c - x) - F(d - x))
            R1L[j, i] /= (4 * pi * constants.epsilon_0 * constants.c * dx * navg)
            if i == 0 or i == n - 1: R1L[j, i] *= 2
    return R1L

def computeR1T(n, dx, a, navg):
    R1T = zeros((n - 1, n - 1))
    for i in range(n - 1):
        c = i * dx
        d = (i + 1) * dx
        for j in range(n - 1):
            for k in range(navg):
                x = (j + k / (navg - 1.0)) * dx
                F = lambda x: log(sqrt(x**2 + a**2) + x)
                R1T[j, i] += (F(d - x) - F(c - x))
            R1T[j, i] /= (4 * pi * constants.epsilon_0 * (constants.c ** 2) * navg)
    return R1T

def getMatrixM(n, dx, dt, R, R2, R1L, R1T):
    M = zeros((n - 1, n - 1))
    for j in range(n - 1):
        ipos = (j + 0.5) * dx
        for i in range(n):
            qpos = min(max(0.25 * dx, i * dx), (n - 1.25) * dx)
            stepsaway = abs((qpos - ipos)) / constants.c / dt
            # R1L term
            if stepsaway < 1:
                if i > 0: M[j, i - 1] += R1L[j, i] * (1 - stepsaway)
                if i < n - 1: M[j, i] -= R1L[j, i] * (1 - stepsaway)
             
            # R2 term
            if stepsaway < 1:
                if i > 0: M[j, i - 1] += 0.5 * ((1 - stepsaway) * dt) ** 2 / dt * R2[j, i]
                if i < n - 1: M[j, i] -= 0.5 * ((1 - stepsaway) * dt) ** 2 / dt * R2[j, i]
            
        for i in range(n - 1):
            # R1T term
            i2pos = (i + 0.5) * dx
            stepsaway = abs((ipos - i2pos)) / constants.c / dt
            if stepsaway < 1: M[j, i] -= R1T[j, i] / dt
    return M - R * identity(n - 1)

def getMatrixE(n, m):
    return concatenate((identity(n - 1), zeros(((m - 1) * (n - 1) + n + 1, n - 1))))

def getMatrixD(n, m):
    D = zeros((m * (n - 1) + n + 1, m * (n - 1) + n + 1))
    D[:n-1,:n-1] = identity(n - 1)
    return D

def getMatrixS(n, m):
    S = zeros((m * (n - 1) + n + 1, m * (n - 1) + n + 1))
    S[n-1:m*(n-1), :(m-1)*(n-1)] = identity((m - 1) * (n - 1))
    S[m*(n-1):, m*(n-1):] = identity(n + 1)
    return S

def getMatrixQ(n, m, dt):
    Q = zeros((m * (n - 1) + n + 1, m * (n - 1) + n + 1))
    Q[m*(n-1):m*(n-1)+n-1, (m-2)*(n-1):(m-1)*(n-1)] -= 0.5 * identity(n - 1)
    Q[m*(n-1)+1:m*(n-1)+n, (m-2)*(n-1):(m-1)*(n-1)] += 0.5 * identity(n - 1)
    Q[m*(n-1):m*(n-1)+n-1, (m-1)*(n-1):m*(n-1)] -= 0.5 * identity(n - 1)
    Q[m*(n-1)+1:m*(n-1)+n, (m-1)*(n-1):m*(n-1)] += 0.5 * identity(n - 1)
    return Q * dt

def getMatrixA(n, m, dx, dt, E_app, R2, R1L, R1T):
    A = zeros((n - 1, m * (n - 1) + n + 1))
    for j in range(n - 1):
        ipos = (j + 0.5) * dx
        for i in range(n):
            qpos = min(max(0.25 * dx, i * dx), (n - 1.25) * dx)
            stepsaway = abs((qpos - ipos)) / constants.c / dt
            # R1L term
            if stepsaway >= 1:
                if i > 0: A[j, (int(stepsaway) - 1) * (n - 1) + i - 1] += R1L[j, i] * (ceil(stepsaway) - stepsaway)
                if i < n - 1: A[j, (int(stepsaway) - 1) * (n - 1) + i] -= R1L[j, i] * (ceil(stepsaway) - stepsaway)
            if i > 0: A[j, int(stepsaway) * (n - 1) + i - 1] += R1L[j, i] * (stepsaway - floor(stepsaway))
            if i < n - 1: A[j, int(stepsaway) * (n - 1) + i] -= R1L[j, i] * (stepsaway - floor(stepsaway))
             
            # R2 term
            deltat = (ceil(stepsaway) - stepsaway) * dt
            if stepsaway >= 1:
                if i > 0: A[j, (ceil(stepsaway) - 2) * (n - 1) + i - 1] += 0.5 * deltat ** 2 / dt * R2[j, i]
                if i < n - 1: A[j, (ceil(stepsaway) - 2) * (n - 1) + i] -= 0.5 * deltat ** 2 / dt * R2[j, i]
            if i > 0: A[j, (ceil(stepsaway) - 1) * (n - 1) + i - 1] += 0.5 * (2 - deltat / dt) * deltat * R2[j, i]
            if i < n - 1: A[j, (ceil(stepsaway) - 1) * (n - 1) + i] -= 0.5 * (2 - deltat / dt) * deltat * R2[j, i]
            A[j, m * (n - 1) + i] += R2[j, i]
            for s in range(int(ceil(stepsaway)) - 1, m - 1):
                if i > 0:
                    A[j, s * (n - 1) + i - 1] += 0.5 * dt * R2[j, i]
                    A[j, (s + 1) * (n - 1) + i - 1] += 0.5 * dt * R2[j, i]
                if i < n - 1:
                    A[j, s * (n - 1) + i] -= 0.5 * dt * R2[j, i]
                    A[j, (s + 1) * (n - 1) + i] -= 0.5 * dt * R2[j, i]
            
        for i in range(n - 1):
            # R1T term
            i2pos = (i + 0.5) * dx
            stepsaway = abs((ipos - i2pos)) / constants.c / dt
            if stepsaway >= 1: A[j, (int(stepsaway) - 1) * (n - 1) + i] -= R1T[j, i] / dt
            A[j, int(stepsaway) * (n - 1) + i] += R1T[j, i] / dt
    A[:, -1] += E_app
    return -A

def getTransitionMatrix(n, m, dx, dt, E_app, R, R2, R1L, R1T):
    M = getMatrixM(n, dx, dt, R, R2, R1L, R1T)
    A = getMatrixA(n, m, dx, dt, E_app, R2, R1L, R1T)
    S = getMatrixS(n, m)
    E = getMatrixE(n, m)
    Q = getMatrixQ(n, m, dt)
    return S + E.dot(linalg.inv(M)).dot(A) + Q

def getTwoStageMatrix(n, m, dx, dt, E_app, R, R2, R1L, R1T):
    M = getMatrixM(n, dx, dt, R, R2, R1L, R1T)
    A = getMatrixA(n, m, dx, dt, E_app, R2, R1L, R1T)
    S = getMatrixS(n, m)
    E = getMatrixE(n, m)
    Q = getMatrixQ(n, m, dt)
    D = getMatrixD(n, m)
    W = linalg.inv(M)
    G1 = S + E.dot(W).dot(A) + Q
    return S + Q + 0.25 * D + 0.25 * E.dot(W).dot(A).dot(2 * identity(m * (n - 1) + n + 1) + G1)
