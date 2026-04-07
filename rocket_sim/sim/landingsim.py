from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rocket_sim.control.landing_controller import LandingController, default_landing_controller
from rocket_sim.sim.physics import ControlOutput, RocketParams, RocketState, step_physics


@dataclass(frozen=True)
class LandingTolerances:
    # Horizontal position accuracy at touchdown.
    x_error_m_abs: float = 5.0
    # Touchdown velocity constraints (soft landing).
    vx_abs_mps: float = 6.0
    vy_abs_mps: float = 4.0
    speed_abs_mps: float = 12.0
    # Require downward motion at touchdown (vy should be negative or near zero).
    require_descending: bool = True


@dataclass
class SimulationResult:
    # Overall status
    success: bool
    failure_reason: str

    # Touchdown metrics (if we touched down)
    touchdown_time_s: Optional[float] = None
    touchdown_x_m: Optional[float] = None
    touchdown_vx_mps: Optional[float] = None
    touchdown_vy_mps: Optional[float] = None
    touchdown_speed_mps: Optional[float] = None
    touchdown_fuel_used_kg: Optional[float] = None

    # Trajectory history for visualization
    history: Dict[str, List[float]] = None


def _touchdown_check(
    touchdown_x_m: float,
    touchdown_vx_mps: float,
    touchdown_vy_mps: float,
    touchdown_speed_mps: float,
    target_x_m: float,
    tol: LandingTolerances,
) -> Tuple[bool, str]:
    x_err = abs(touchdown_x_m - target_x_m)
    if x_err > tol.x_error_m_abs:
        return False, f"x error too large ({x_err:.2f} m)"
    if abs(touchdown_vx_mps) > tol.vx_abs_mps:
        return False, f"vx too large ({touchdown_vx_mps:.2f} m/s)"
    if abs(touchdown_vy_mps) > tol.vy_abs_mps:
        return False, f"vy too large ({touchdown_vy_mps:.2f} m/s)"
    if touchdown_speed_mps > tol.speed_abs_mps:
        return False, f"speed too large ({touchdown_speed_mps:.2f} m/s)"
    if tol.require_descending and touchdown_vy_mps > 0.5:  # allow tiny upward due to interpolation
        return False, "not descending at touchdown"
    return True, "landing successful"


def simulate_landing(
    *,
    params: RocketParams,
    target_x_m: float,
    initial_height_m: float,
    initial_fuel_mass_kg: float,
    controller: Optional[LandingController] = None,
    dt_s: float = 0.02,
    max_time_s: float = 120.0,
    tolerances: LandingTolerances = LandingTolerances(),
) -> SimulationResult:
    if initial_height_m <= 0.0:
        return SimulationResult(success=False, failure_reason="initial height must be > 0", history={})
    if initial_fuel_mass_kg < 0.0:
        return SimulationResult(success=False, failure_reason="initial fuel must be >= 0", history={})

    if controller is None:
        controller = default_landing_controller(params)
    controller.reset()

    state = RocketState(
        x_m=0.0,
        y_m=float(initial_height_m),
        vx_mps=0.0,
        vy_mps=0.0,
        fuel_mass_kg=float(initial_fuel_mass_kg),
        t_s=0.0,
    )

    fuel_initial = state.fuel_mass_kg

    steps = max(1, int(max_time_s / dt_s))

    # History arrays for visualization.
    history: Dict[str, List[float]] = {
        "t_s": [],
        "x_m": [],
        "y_m": [],
        "vx_mps": [],
        "vy_mps": [],
        "thrust_N": [],
        "gimbal_deg": [],
        "fuel_mass_kg": [],
    }

    touchdown_found = False
    touchdown_metrics = {}

    for i in range(steps):
        t_prev = state.t_s
        state_prev = RocketState(
            x_m=state.x_m,
            y_m=state.y_m,
            vx_mps=state.vx_mps,
            vy_mps=state.vy_mps,
            fuel_mass_kg=state.fuel_mass_kg,
            t_s=state.t_s,
        )

        control: ControlOutput = controller.compute_control(state, target_x_m=target_x_m, dt_s=dt_s)
        step = step_physics(state, params, control, dt_s)
        state = step.next_state

        # Record history (after step).
        history["t_s"].append(state.t_s)
        history["x_m"].append(state.x_m)
        history["y_m"].append(state.y_m)
        history["vx_mps"].append(state.vx_mps)
        history["vy_mps"].append(state.vy_mps)
        history["thrust_N"].append(step.actual_thrust_N)
        history["gimbal_deg"].append(step.actual_gimbal_angle_deg)
        history["fuel_mass_kg"].append(state.fuel_mass_kg)

        if state.y_m <= 0.0:
            # Interpolate to y=0 for better touchdown metrics.
            if state_prev.y_m > 0.0:
                ratio = state_prev.y_m / (state_prev.y_m - state.y_m)  # in (0,1]
            else:
                ratio = 0.0

            touchdown_time_s = t_prev + ratio * dt_s
            touchdown_x_m = state_prev.x_m + (state.x_m - state_prev.x_m) * ratio
            touchdown_vx_mps = state_prev.vx_mps + (state.vx_mps - state_prev.vx_mps) * ratio
            touchdown_vy_mps = state_prev.vy_mps + (state.vy_mps - state_prev.vy_mps) * ratio
            touchdown_speed_mps = math.sqrt(touchdown_vx_mps**2 + touchdown_vy_mps**2)
            fuel_touch = state_prev.fuel_mass_kg + (state.fuel_mass_kg - state_prev.fuel_mass_kg) * ratio
            touchdown_fuel_used_kg = fuel_initial - fuel_touch

            success, reason = _touchdown_check(
                touchdown_x_m=touchdown_x_m,
                touchdown_vx_mps=touchdown_vx_mps,
                touchdown_vy_mps=touchdown_vy_mps,
                touchdown_speed_mps=touchdown_speed_mps,
                target_x_m=target_x_m,
                tol=tolerances,
            )

            touchdown_found = True
            touchdown_metrics = dict(
                touchdown_time_s=touchdown_time_s,
                touchdown_x_m=touchdown_x_m,
                touchdown_vx_mps=touchdown_vx_mps,
                touchdown_vy_mps=touchdown_vy_mps,
                touchdown_speed_mps=touchdown_speed_mps,
                touchdown_fuel_used_kg=touchdown_fuel_used_kg,
                success=success,
                failure_reason=reason if not success else "landing successful",
            )
            break

        # Hard failure if we run out of fuel while still in the air.
        if state.fuel_mass_kg <= 0.0 and state.y_m > 0.0:
            return SimulationResult(
                success=False,
                failure_reason="out of fuel before touchdown",
                history=history,
            )

    if not touchdown_found:
        return SimulationResult(
            success=False,
            failure_reason="time limit reached before touchdown",
            history=history,
        )

    return SimulationResult(success=touchdown_metrics["success"], failure_reason=touchdown_metrics["failure_reason"], history=history, **{k: v for k, v in touchdown_metrics.items() if k.startswith("touchdown_")})

