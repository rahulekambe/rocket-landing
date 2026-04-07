from __future__ import annotations

import math
from dataclasses import dataclass

from rocket_sim.control.pid import PID, PIDConfig
from rocket_sim.sim.physics import ControlOutput, RocketParams, RocketState, wind_x_accel_mps2


@dataclass(frozen=True)
class LandingControllerConfig:
    # PID tuning
    pid_x: PIDConfig
    pid_y: PIDConfig

    # Vertical guidance mode.
    # - "tgo": physics-friendly braking using a time-to-go estimate (works best when vy is negative).
    # - "vy_schedule": continuously command a desired downward velocity based on altitude.
    # - "pid": use pid_y directly on height error.
    vertical_mode: str = "vy_schedule"
    min_t_go_s: float = 0.3
    max_t_go_s: float = 8.0
    ay_total_limit_abs: float = 60.0
    v_des_max_mps: float = 25.0
    y_ref_m: float = 600.0
    vertical_kp: float = 2.0

    # Actuator constraints
    max_thrust_rate_N_per_s: float = 15000.0
    max_gimbal_rate_deg_per_s: float = 90.0

    # If true, estimate the wind term and compensate in horizontal thrust direction.
    compensate_wind: bool = True


class LandingController:
    """
    Autopilot that chooses thrust magnitude + gimbal angle to land at `target_x`.

    Control philosophy:
      1. Use PID on horizontal position error to produce desired *total* ax.
      2. Use either a physics-friendly braking law or PID on vertical height to produce desired *total* ay.
      3. Convert desired (ax, ay) into thrust direction and magnitude with gimbal constraints.
      4. Apply rate limiting to prevent oscillations.
    """

    def __init__(self, params: RocketParams, config: LandingControllerConfig):
        self.params = params
        self.config = config

        self.pid_x = PID(config.pid_x)
        self.pid_y = PID(config.pid_y)

        self._prev_thrust_N = 0.0
        self._prev_angle_deg = 0.0
        self._prev_t_s = None

    def reset(self) -> None:
        self.pid_x.reset()
        self.pid_y.reset()
        self._prev_thrust_N = 0.0
        self._prev_angle_deg = 0.0
        self._prev_t_s = None

    def _rate_limit(self, desired: float, prev: float, rate_limit: float, dt_s: float) -> float:
        if dt_s <= 0.0:
            return desired
        max_delta = rate_limit * dt_s
        return max(prev - max_delta, min(prev + max_delta, desired))

    def compute_control(self, state: RocketState, target_x_m: float, dt_s: float) -> ControlOutput:
        """
        Produce ControlOutput for the next physics step.

        Note: dt_s is assumed to match the physics step dt_s for stable derivatives.
        """
        t_s = state.t_s
        dt = max(dt_s, 1e-6)

        # Horizontal: want x -> target_x and vx -> 0 implicitly via derivative term.
        x_err = target_x_m - state.x_m  # positive when rocket is left of target
        ax_total_des = self.pid_x.update(x_err, dt)

        # Vertical guidance.
        if self.config.vertical_mode == "tgo":
            # Estimate a time-to-go that would (approximately) bring vy to 0 at touchdown
            # under constant vertical acceleration.
            #
            # Derived (with constant acceleration) from:
            #   0 = y + vy*t + 0.5*a*t^2
            #   0 = vy + a*t    => a = -vy/t
            # => t_go ~= -2*y/vy   (valid when vy is sufficiently negative)
            y = max(0.0, state.y_m)
            vy = state.vy_mps

            if vy < -0.5:
                t_go = -2.0 * y / vy
            else:
                # If not descending fast, fall back to free-fall time scale.
                t_go = math.sqrt(max(1e-6, 2.0 * y / self.params.g_mps2))

            t_go = max(self.config.min_t_go_s, min(self.config.max_t_go_s, t_go))
            ay_total_des = -vy / t_go  # braking: when vy is negative, this is positive
            ay_total_des = max(-self.config.ay_total_limit_abs, min(self.config.ay_total_limit_abs, ay_total_des))
        elif self.config.vertical_mode == "vy_schedule":
            # Command a desired downward velocity based on current altitude, then
            # use it to set desired total vertical acceleration.
            #
            # vy_des is negative when y>0, so the rocket keeps descending (avoids hover lock).
            y = max(0.0, state.y_m)
            vy = state.vy_mps
            ratio = 0.0 if self.config.y_ref_m <= 0.0 else max(0.0, min(1.0, y / self.config.y_ref_m))
            vy_des = -self.config.v_des_max_mps * math.sqrt(ratio)
            ay_total_des = self.config.vertical_kp * (vy_des - vy)
            ay_total_des = max(-self.config.ay_total_limit_abs, min(self.config.ay_total_limit_abs, ay_total_des))
        else:
            # PID mode on height (error = -y).
            y_err = -state.y_m
            ay_total_des = self.pid_y.update(y_err, dt)

        # Wind compensation: if the simulation model adds ax_wind, subtract it to find thrust contribution.
        wind_ax = 0.0
        if self.config.compensate_wind:
            wind_ax = wind_x_accel_mps2(t_s, state, self.params)

        ax_thrust_des = ax_total_des - wind_ax
        ay_thrust_des = ay_total_des + self.params.g_mps2  # because ay_total = (T/m)cos(theta) - g

        # Thrust can only "push"; if the controller asks for negative vertical thrust, we drop it.
        ay_thrust_des = max(0.0, ay_thrust_des)

        m = self.params.dry_mass_kg + max(0.0, state.fuel_mass_kg)
        if m <= 0.0:
            desired_thrust = 0.0
            desired_angle_deg = 0.0
        else:
            # Convert desired thrust components into angle + magnitude.
            # ax_thrust = (T/m) * sin(theta)
            # ay_thrust = (T/m) * cos(theta)
            desired_angle_rad = math.atan2(ax_thrust_des, max(1e-9, ay_thrust_des))
            desired_angle_deg = math.degrees(desired_angle_rad)
            desired_angle_deg = max(-self.params.max_gimbal_deg, min(self.params.max_gimbal_deg, desired_angle_deg))

            # Choose thrust magnitude primarily to match vertical component (stable near landing).
            cos_theta = math.cos(math.radians(desired_angle_deg))
            if abs(cos_theta) < 1e-6 or ay_thrust_des <= 0.0:
                desired_thrust = 0.0
            else:
                desired_thrust = (m * ay_thrust_des) / cos_theta

        desired_thrust = max(0.0, min(self.params.max_thrust_N, desired_thrust))

        # Apply rate limiting (helps keep landing "smooth").
        desired_thrust = self._rate_limit(
            desired_thrust, self._prev_thrust_N, self.config.max_thrust_rate_N_per_s, dt
        )
        desired_angle_deg = self._rate_limit(
            desired_angle_deg, self._prev_angle_deg, self.config.max_gimbal_rate_deg_per_s, dt
        )

        self._prev_thrust_N = desired_thrust
        self._prev_angle_deg = desired_angle_deg
        self._prev_t_s = t_s

        return ControlOutput(thrust_N=desired_thrust, gimbal_angle_deg=desired_angle_deg)


def default_landing_controller(params: RocketParams) -> LandingController:
    """
    Reasonable default PID gains chosen for the default simulator parameter ranges.

    If you expose UI controls to change physics settings dramatically, you may need to retune.
    """
    # Horizontal: PID produces desired total ax.
    pid_x = PIDConfig(kp=0.08, ki=0.0, kd=0.18, output_limit_abs=35.0, integral_limit_abs=250.0)

    # Vertical: default uses the "tgo" guidance law, but we keep pid_y for "pid" mode.
    pid_y = PIDConfig(kp=0.02, ki=0.0, kd=2.00, output_limit_abs=60.0, integral_limit_abs=600.0)

    cfg = LandingControllerConfig(
        pid_x=pid_x,
        pid_y=pid_y,
        vertical_mode="vy_schedule",
        min_t_go_s=0.3,
        max_t_go_s=8.0,
        ay_total_limit_abs=60.0,
    )
    return LandingController(params=params, config=cfg)

