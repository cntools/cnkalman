# CNKalman

This is a relatively low level implementation of a kalman filter; with support for extended and iterative extended
kalman filters. The goals of the project are to provide a numerically stable, robust EKF implementation which is both
fast and portable. 

The main logic is written in C and only needs the associated matrix library to work; and there are C++ wrappers provided 
for convenience.