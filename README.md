# Rocket Booster Landing Simulation
This project simulates a **2D rocket** descending from an initial height to a **target landing zone** 

## Streamlit UI

Run the app and adjust:

- `Initial height (m)` – starting `y`
- `Fuel (kg)` – initial remaining propellant
- `Target position x (m)` – desired landing-zone x coordinate
- `Advanced (wind)` settings – wind amplitude, frequency, and wind drag coefficient
- Visualization:
  - trajectory plot (up to a selected animation frame)
  - moving rocket marker via the **Animation frame** slider
  - `Live animation` to preview the descent quickly

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

