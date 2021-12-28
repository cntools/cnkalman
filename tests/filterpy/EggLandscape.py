from math import cos, sin, tan, atan2

import numpy as np
import sympy
from filterpy.kalman import ExtendedKalmanFilter as EKF
from numpy import array, sqrt
from sympy import symbols, Matrix

class EggLandscape(EKF):
    def __init__(self):
        EKF.__init__(self, 4, 3)
        self.Q = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, .0001, 0],
            [0, 0, 0, 0.0001],
        ])
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            ])

    def predict_meas(self, x, idx):
        landmark_pos = self.landmarks[idx]
        px = landmark_pos[0]
        py = landmark_pos[1]
        dist = sqrt((px - x[0, 0])**2 + (py - x[1, 0])**2)

        Hx = array([[dist],
                    [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])
        return Hx

    def update(self, z, R):
        var_a = 101
        var_b = 11
        coeff_dist = 5;

        def HJacobian(xs):
            x = xs[0][0]
            y = xs[1][0]
            c = cos(x * x + y * y)

            return np.array([
                [2 * x * coeff_dist * c, 2 * y * coeff_dist * c, 0, 0],
                [var_a*cos(var_a*x)*sin(y*var_a), var_a*cos(var_a*y)*sin(x*var_a), 0, 0],
                [var_b*cos(var_b*x)* cos(y * var_b), -var_b*sin(var_b*y)*sin(x * var_b), 0, 0],
            ])

        def Hx(x):
            return np.array([
                sin((x[0] * x[0] + x[1] * x[1]) * coeff_dist),
                sin(x[0] * var_a) * sin(x[1] * var_a),
                sin(x[0] * var_b) * cos(x[1] * var_b)
            ]).reshape(-1,1)
        EKF.update(self, z, HJacobian=HJacobian, Hx=Hx, R=R)