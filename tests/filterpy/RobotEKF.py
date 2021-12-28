from math import cos, sin, tan, atan2

import numpy as np
import sympy
from filterpy.kalman import ExtendedKalmanFilter as EKF
from numpy import array, sqrt
from sympy import symbols, Matrix


class RobotEKF(EKF):
    def __init__(self, dt, wheelbase, u, std_vel, std_steer):
        EKF.__init__(self, 3, 2, 2)
        self.u = u
        self.dt = dt
        self.wheelbase = wheelbase
        self.std_vel = std_vel
        self.std_steer = std_steer
        self.landmarks = array([[5, 10], [10, 5], [15, 15]])

        a, x, y, v, w, theta, time = symbols(
            'a, x, y, v, w, theta, t')
        d = v*time
        beta = (d/w)*sympy.tan(a)
        r = w/sympy.tan(a)

        self.fxu = Matrix(
            [[x-r*sympy.sin(theta)+r*sympy.sin(theta+beta)],
             [y+r*sympy.cos(theta)-r*sympy.cos(theta+beta)],
             [theta+beta]])

        self.F_j = self.fxu.jacobian(Matrix([x, y, theta]))
        self.V_j = self.fxu.jacobian(Matrix([v, a]))

        # save dictionary and it's variables for later use
        self.subs = {x: 0, y: 0, v:0, a:0,
                     time:dt, w:wheelbase, theta:0}
        self.x_x, self.x_y, = x, y
        self.v, self.a, self.theta = v, a, theta

    def predict(self, u=None):
        if u is None:
            u = self.u
        self.x = self.move(self.x, u, self.dt)

        self.subs[self.theta] = self.x[2, 0]
        self.subs[self.v] = u[0]
        self.subs[self.a] = u[1]

        F = array(self.F_j.evalf(subs=self.subs)).astype(float)
        V = array(self.V_j.evalf(subs=self.subs)).astype(float)

        # covariance of motion noise in control space
        M = array([[self.std_vel*u[0]**2, 0],
                   [0, self.std_steer**2]])

        self.P = F @ self.P @ F.T + V @ M @ V.T

    def move(self, x, u, dt):
        hdg = x[2, 0]
        vel = u[0]
        steering_angle = u[1]
        dist = vel * dt

        if abs(steering_angle) > 0.001: # is robot turning?
            beta = (dist / self.wheelbase) * tan(steering_angle)
            r = self.wheelbase / tan(steering_angle) # radius

            dx = np.array([[-r*sin(hdg) + r*sin(hdg + beta)],
                           [r*cos(hdg) - r*cos(hdg + beta)],
                           [beta]])
        else: # moving in straight line
            dx = np.array([[dist*cos(hdg)],
                           [dist*sin(hdg)],
                           [0]])
        return x + dx

    def predict_meas(self, x, idx):
        landmark_pos = self.landmarks[idx]
        px = landmark_pos[0]
        py = landmark_pos[1]
        dist = sqrt((px - x[0, 0])**2 + (py - x[1, 0])**2)

        Hx = array([[dist],
                    [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])
        return Hx

    def update(self, z, R, landmark_idx):
        def residual(a, b):
            """ compute residual (a-b) between measurements containing
            [range, bearing]. Bearing is normalized to [-pi, pi)"""
            y = a - b
            y[1] = y[1] % (2 * np.pi)    # force in range [0, 2 pi)
            if y[1] > np.pi:             # move to [-pi, pi)
                y[1] -= 2 * np.pi
            return y

        def Hx(x, landmark_pos):
            """ takes a state variable and returns the measurement
            that would correspond to that state.
            """
            px = landmark_pos[0]
            py = landmark_pos[1]
            dist = sqrt((px - x[0, 0])**2 + (py - x[1, 0])**2)

            Hx = array([[dist],
                        [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])
            return Hx
        def H_of(x, landmark_pos):
            """ compute Jacobian of H matrix where h(x) computes
            the range and bearing to a landmark for state x """

            px = landmark_pos[0]
            py = landmark_pos[1]
            hyp = (px - x[0, 0])**2 + (py - x[1, 0])**2
            dist = sqrt(hyp)

            H = array(
                [[-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
                 [ (py - x[1, 0]) / hyp,  -(px - x[0, 0]) / hyp, -1]])
            return H
        landmark = self.landmarks[landmark_idx]

        EKF.update(self, z, HJacobian=H_of, Hx=Hx, R=R,
               residual=residual,
               args=(landmark), hx_args=(landmark))