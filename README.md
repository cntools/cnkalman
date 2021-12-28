# CNKalman [![Build and Test](https://github.com/cntools/cnkalman/actions/workflows/cmake.yml/badge.svg)](https://github.com/cntools/cnkalman/actions/workflows/cmake.yml)

This is a relatively low level implementation of a kalman filter; with support for extended and iterative extended
kalman filters. The goals of the project are to provide a numerically stable, robust EKF implementation which is both
fast and portable. 

The main logic is written in C and only needs the associated matrix library to work; and there are C++ wrappers provided 
for convenience.

## Features

- Support for [extended kalman filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter), [linear kalman filters](https://en.wikipedia.org/wiki/Kalman_filter), and [Iterate Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter#Iterated_extended_Kalman_filter) ([paper](https://www.diva-portal.org/smash/get/diva2:844060/FULLTEXT01.pdf))
- Support for [adaptive measurement covariance](https://arxiv.org/pdf/1702.00884.pdf)
- Built-in support for numerical-based jacobians, and an option to debug user provided jacobians by using 
  the numerical results
- Supports multiple measurement models per filter, which can be integrated at varying frequencies
