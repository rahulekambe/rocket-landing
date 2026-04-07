from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIDConfig:
    kp: float
    ki: float
    kd: float
    output_limit_abs: float
    integral_limit_abs: float


class PID:
    """
    Classic PID controller with:
      - derivative computed from error delta / dt
      - clamped integral to reduce windup
      - output clamping
    """

    def __init__(self, config: PIDConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._prev_error = None
        self._integral = 0.0

    def update(self, error: float, dt_s: float) -> float:
        if dt_s <= 0:
            return 0.0

        if self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / dt_s

        self._integral += error * dt_s
        # Clamp integral term for stability.
        self._integral = max(-self.config.integral_limit_abs, min(self.config.integral_limit_abs, self._integral))

        output = (
            self.config.kp * error
            + self.config.ki * self._integral
            + self.config.kd * derivative
        )
        output = max(-self.config.output_limit_abs, min(self.config.output_limit_abs, output))

        self._prev_error = error
        return output

