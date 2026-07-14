# Navigation Algorithms

The following are key functions related to navigation algorithms.

## MagNav Filter Model

```@docs
MagNav.create_model
```

### MagNav filter model internals

```@docs
MagNav.create_P0
```

```@docs
MagNav.create_Qd
```

```@docs
MagNav.get_pinson
```

## Cramér–Rao Lower Bound

```@docs
MagNav.crlb
```

## Extended Kalman Filter

```@docs
MagNav.ekf
```

## Factor Graph Optimization

The factor graph optimization (FGO) formulation poses navigation as a batch
maximum a posteriori (MAP) estimation problem over the full flight. The error
states are the variables, and prior, process (motion), and magnetic measurement
factors define the objective. Because the Pinson error model is a linear-Gaussian
chain, the MAP estimate is obtained exactly with an iterated fixed-interval
(Rauch–Tung–Striebel) smoother that reuses the same model as [`MagNav.ekf`](@ref).
Unlike the causal EKF, every estimate is informed by all measurements (past and
future), which generally reduces navigation error.

```@docs
MagNav.fgo
```

### FGO with batch Tolles-Lawson estimation

Aeromagnetic compensation (Tolles-Lawson) coefficients can be added to the
factor graph as variables, so the platform field calibration is estimated
jointly with the navigation states over the whole flight (the batch analog of
[`MagNav.ekf_online`](@ref)).

```@docs
MagNav.fgo_online
```

### FGO with scalar magnetometer sensor-error factors

Scalar magnetometer sensor errors — optically-pumped / quantum heading error
(modeled physically as a Fourier series in the sensor–field angle `θ`: vector
light shift `∝ cos θ` and nonlinear Zeeman `∝ cos 2θ`), sensor dead zones
(`∝ 1/|sin 2θ|` noise inflation), fluxgate hard-iron bias, and electronics
drift — can be added to the factor graph as variables and estimated jointly with
navigation. A low-dimensional grid or point-mass estimator (which carries only
position) cannot do this; the continuous factor graph recovers a sensor
calibration as a by-product.

```@docs
MagNav.fgo_sensor
```

## Run Filter (with additional options)

```@docs
MagNav.run_filt
```
