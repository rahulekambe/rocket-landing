# Rocket Simulation ( with Streamlit)

This project simulates a **2D rocket** descending from an initial height to a **target landing zone** (a desired x-position on the ground). It includes:

- **Physics**: gravity, thrust with gimbal angle, and **fuel consumption**
- **Disturbance**: a simple **wind gust** model causing lateral acceleration
- **Autopilot** (AI control logic): an onboard controller that chooses **thrust magnitude** and **gimbal angle** to land smoothly
- **Streamlit UI**: interactive inputs + real-time visualization and landing metrics

## How the simulation works

### 1) Rocket dynamics 
The rocket is modeled as a **point mass** in 2D with state:

`x` (horizontal position), `y` (height above ground), `vx`, `vy`, and remaining `fuel`.

Each simulation step advances time using a semi-implicit Euler update:

- **Gravity** acts downward: `ay -= g`
- **Thrust** points along the gimbaled direction:
  - `ax_thrust = (T/m) * sin(theta)`
  - `ay_thrust = (T/m) * cos(theta)`
  - where `theta` is the gimbal angle relative to vertical
- **Wind disturbance** adds lateral acceleration:
  - a proportional model nudges the rocket toward the current wind velocity

#### Fuel-limited thrust
Even if the controller requests thrust `T`, the engine is constrained by remaining fuel. If fuel would run out mid-step, thrust is reduced so consumption stays physically consistent for that `dt`.

### 2) Wind model
Wind is a sinusoidal gust:

- `wind_v(t) = A * sin(2π f t + phase)`
- lateral acceleration behaves like: `ax_wind = k * (wind_v - vx)`

This keeps the disturbance realistic enough to test controller stability.

### 3) Touchdown detection 
The simulation runs until `y <= 0` (ground). When touchdown happens, the code interpolates the final step to estimate:

- touchdown time
- touchdown position `x`
- touchdown velocities (`vx`, `vy`) and speed
- fuel used

Landing is considered **successful** only if all tolerances pass:

- `|x_error| <= 5 m`
- touchdown `vx`, `vy`, and total speed are below limits
- rocket must be descending at touchdown (`vy` not significantly positive)

## How the autopilot controls the landing

The autopilot lives in `rocket_sim/control/landing_controller.py` and produces a `ControlOutput`:

- `thrust_N` (thrust magnitude)
- `gimbal_angle_deg` (tilt relative to vertical)

### Horizontal guidance (x-axis)
Uses a PID controller on **horizontal position error**:

- error: `x_error = target_x - x`
- controller output is interpreted as desired total horizontal acceleration `ax_total_des`

Wind compensation is applied (the controller subtracts the modeled wind lateral acceleration from its desired thrust contribution).

### Vertical guidance (y-axis)
To avoid hover-lock and still achieve a soft landing, the default vertical logic uses a **velocity-schedule**:

- compute altitude ratio `y / y_ref`
- choose a desired **downward** velocity `vy_des` that depends on altitude
- use proportional control to convert `(vy_des - vy)` into desired total vertical acceleration `ay_total_des`

This keeps the rocket descending while ramping up braking near the ground.

### Converting (ax, ay) to thrust + angle
Given desired thrust components, the controller:

1. Computes desired gimbal angle from `ax_thrust_des` and `ay_thrust_des`
2. Computes thrust magnitude from the vertical component (stable near landing)
3. Applies **rate limiting** to keep changes smooth
4. Clamps thrust and gimbal to physical limits

## Streamlit UI

Run the app and adjust:

- `Initial height (m)` – starting `y`
- `Fuel (kg)` – initial remaining propellant
- `Target position x (m)` – desired landing-zone x coordinate
- Optional `Advanced (wind)` settings – wind amplitude, frequency, and wind drag coefficient
- Visualization:
  - trajectory plot (up to a selected animation frame)
  - moving rocket marker via the **Animation frame** slider
  - optional `Live animation` to preview the descent quickly

## Metrics shown after a run

- **Landing success/failure**
- **Velocity at touchdown** (speed, `vx`, `vy` implied via speed)
- **Fuel used** (kg)
- **X error at touchdown** (meters)

## Project structure

- `app.py` - Streamlit UI and visualization
- `rocket_sim/sim/physics.py` - rocket physics step (gravity, thrust, fuel, wind)
- `rocket_sim/sim/landingsim.py` - simulation loop + touchdown interpolation + scoring
- `rocket_sim/control/pid.py` - PID implementation used by the controller
- `rocket_sim/control/landing_controller.py` - AI landing controller (thrust + gimbal logic)

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Visualization uses `matplotlib`. If the environment is missing it, the UI may not render plots (the simulation logic itself still works).

