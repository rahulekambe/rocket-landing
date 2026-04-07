from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RocketParams:
    # Mass model
    dry_mass_kg: float = 500.0

    # Propulsion model
    isp_s: float = 300.0
    max_thrust_N: float = 25000.0
    max_gimbal_deg: float = 20.0  # gimbal angle relative to vertical

    # Gravity
    g_mps2: float = 9.81
    g0_mps2: float = 9.81  # keep naming explicit for rocket equation units

    # Wind disturbance model
    wind_amp_mps: float = 10.0  # wind velocity amplitude (m/s)
    wind_freq_hz: float = 0.05  # gust frequency
    wind_phase_rad: float = 0.0
    wind_drag_coeff_1_per_s: float = 0.05  # scales (wind_v - vx) into lateral acceleration


@dataclass
class RocketState:
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    fuel_mass_kg: float
    t_s: float


@dataclass(frozen=True)
class ControlOutput:
    thrust_N: float
    gimbal_angle_deg: float  # relative to vertical; + means thrust tilts right


@dataclass(frozen=True)
class StepResult:
    next_state: RocketState
    actual_thrust_N: float
    actual_gimbal_angle_deg: float


def wind_velocity_mps(t_s: float, params: RocketParams) -> float:
    """Simple sinusoidal gust wind velocity (m/s) along x."""
    return params.wind_amp_mps * math.sin(2.0 * math.pi * params.wind_freq_hz * t_s + params.wind_phase_rad)


def wind_x_accel_mps2(t_s: float, state: RocketState, params: RocketParams) -> float:
    """
    Approximate lateral acceleration due to gusts.

    This behaves like a simple proportional drag towards the wind velocity:
      ax_wind = k * (wind_v - vx)
    """
    wind_v = wind_velocity_mps(t_s, params)
    return params.wind_drag_coeff_1_per_s * (wind_v - state.vx_mps)


def step_physics(state: RocketState, params: RocketParams, control: ControlOutput, dt_s: float) -> StepResult:
    """
    Advance rocket point-mass state by one time step using semi-implicit Euler.

    Equations:
      ax = (T/m) * sin(theta) + ax_wind
      ay = (T/m) * cos(theta) - g
    Where theta is the gimbal angle relative to vertical.
    """
    thrust_cmd = max(0.0, float(control.thrust_N))
    theta_deg_cmd = float(control.gimbal_angle_deg)

    # Clamp actuator angle.
    theta_deg = max(-params.max_gimbal_deg, min(params.max_gimbal_deg, theta_deg_cmd))
    theta_rad = math.radians(theta_deg)

    fuel_prev = max(0.0, float(state.fuel_mass_kg))
    m_prev = params.dry_mass_kg + fuel_prev

    # If we run out of fuel mid-step, limit thrust so mdot*dt doesn't exceed remaining fuel.
    # mdot = T / (isp * g0)  =>  fuel_used = T * dt / (isp * g0)
    if fuel_prev <= 0.0 or thrust_cmd <= 0.0:
        thrust_eff = 0.0
        fuel_used = 0.0
    else:
        max_thrust_by_fuel = fuel_prev * params.isp_s * params.g0_mps2 / max(dt_s, 1e-9)
        thrust_eff = min(thrust_cmd, max_thrust_by_fuel)
        fuel_used = thrust_eff * dt_s / (params.isp_s * params.g0_mps2)

    fuel_next = max(0.0, fuel_prev - fuel_used)

    # Wind disturbance uses previous state (t and vx before integration).
    ax_wind = wind_x_accel_mps2(state.t_s, state, params)

    # Thrust accelerations (use previous mass for stability).
    if m_prev <= 0.0 or thrust_eff <= 0.0:
        ax_thrust = 0.0
        ay_thrust = 0.0
    else:
        ax_thrust = (thrust_eff / m_prev) * math.sin(theta_rad)
        ay_thrust = (thrust_eff / m_prev) * math.cos(theta_rad)

    ax = ax_thrust + ax_wind
    ay = ay_thrust - params.g_mps2

    # Semi-implicit Euler: update velocity, then position.
    vx_next = state.vx_mps + ax * dt_s
    vy_next = state.vy_mps + ay * dt_s
    x_next = state.x_m + vx_next * dt_s
    y_next = state.y_m + vy_next * dt_s
    t_next = state.t_s + dt_s

    next_state = RocketState(
        x_m=x_next,
        y_m=y_next,
        vx_mps=vx_next,
        vy_mps=vy_next,
        fuel_mass_kg=fuel_next,
        t_s=t_next,
    )
    return StepResult(next_state=next_state, actual_thrust_N=thrust_eff, actual_gimbal_angle_deg=theta_deg)

