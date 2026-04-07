from __future__ import annotations

import time
from typing import Dict

import numpy as np
import streamlit as st

from rocket_sim.sim.landingsim import LandingTolerances, simulate_landing
from rocket_sim.sim.physics import RocketParams

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover (depends on runtime environment)
    plt = None


def _render_trajectory(history: Dict[str, list], target_x_m: float, frame: int):
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for trajectory visualization.")
    x = np.asarray(history["x_m"])
    y = np.asarray(history["y_m"])

    # Scene-like visualization: sky + ground + rocket sprite.
    fig, ax = plt.subplots(figsize=(7, 4.0))
    # Make the whole figure background black (outside the axes as well).
    fig.patch.set_facecolor("black")

    # Overall scene bounds.
    y_max = float(np.max(y)) * 1.1 + 10.0
    x_min = min(float(np.min(x)), float(target_x_m) - 80.0) - 20.0
    x_max = max(float(np.max(x)), float(target_x_m) + 80.0) + 20.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, y_max)
    ax.set_aspect("equal", adjustable="box")

    # Background "sky".
    ax.set_facecolor("#0b1e3b")

    # Ground strip.
    ground_y = 0.0
    ax.fill_between([x_min, x_max], ground_y - 5.0, ground_y + 2.0, color="#3b3b3b")

    # Landing pad at target.
    pad_width = 20.0
    pad_height = 1.0
    pad_x0 = target_x_m - pad_width / 2.0
    ax.add_patch(
        plt.Rectangle(
            (pad_x0, ground_y + 2.0),
            pad_width,
            pad_height,
            facecolor="#4caf50",
            edgecolor="white",
            linewidth=1.0,
        )
    )

    # Rocket body as a small rectangle + triangle nose + simple flame.
    rx = float(x[frame])
    ry = float(y[frame])
    body_h = max(10.0, 0.03 * y_max)
    body_w = body_h / 3.0

    # Body.
    ax.add_patch(
        plt.Rectangle(
            (rx - body_w / 2.0, ry),
            body_w,
            body_h,
            facecolor="#f5f5f5",
            edgecolor="#cccccc",
            linewidth=1.0,
        )
    )

    # Nose.
    nose = plt.Polygon(
        [
            (rx - body_w / 2.0, ry + body_h),
            (rx + body_w / 2.0, ry + body_h),
            (rx, ry + body_h + body_w),
        ],
        closed=True,
        facecolor="#ff7043",
        edgecolor="#ffab91",
        linewidth=1.0,
    )
    ax.add_patch(nose)

    # Flame (only if above ground a bit).
    if ry > ground_y + 1.0:
        flame = plt.Polygon(
            [
                (rx - body_w / 3.0, ry),
                (rx + body_w / 3.0, ry),
                (rx, ry - body_h / 2.5),
            ],
            closed=True,
            facecolor="#ffca28",
            edgecolor="#ffeb3b",
            linewidth=0.8,
            alpha=0.9,
        )
        ax.add_patch(flame)

    # Optional faint trail (up to current frame).
    if frame > 0:
        ax.plot(x[: frame + 1], y[: frame + 1], color="#90caf9", linewidth=1.2, alpha=0.6)

    # Minimal UI text instead of axes.
    ax.axis("off")
    ax.set_title("Landing Scene", color="white", fontsize=13, pad=10)

    return fig


def _render_velocity(history: Dict[str, list]):
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for velocity visualization.")
    t = np.asarray(history["t_s"])
    vx = np.asarray(history["vx_mps"])
    vy = np.asarray(history["vy_mps"])
    speed = np.sqrt(vx**2 + vy**2)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, speed, linewidth=2.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (m/s)")
    ax.set_title("Touchdown dynamics")
    ax.grid(True, alpha=0.35)
    return fig


def main():
    st.set_page_config(page_title="Rocket Booster Landing Simulator", layout="wide")
    st.title("Rocket Booster Landing Simulator")

    st.caption("PID-based autopilot + wind disturbance + fuel-limited thrust. Tune inputs and watch the rocket land at the target zone.")

    with st.sidebar:
        st.header("Mission Setup")
        initial_height_m = st.slider("Initial height (m)", 50, 3000, 1200, step=50)
        initial_fuel_mass_kg = st.slider("Fuel (kg)", 0, 5000, 1200, step=50)
        target_x_m = st.slider("Target position x (m)", -1000, 1000, 200, step=10)

        st.divider()
        show_advanced = st.checkbox("Advanced (wind)", value=True)
        wind_amp_mps = 10.0
        wind_freq_hz = 0.05
        wind_drag_coeff = 0.05
        if show_advanced:
            st.subheader("Wind disturbance")
            wind_amp_mps = st.slider("Wind amplitude (m/s)", 0.0, 30.0, 10.0, step=1.0)
            wind_freq_hz = st.slider("Wind frequency (Hz)", 0.01, 0.2, 0.05, step=0.01)
            wind_drag_coeff = st.slider("Wind drag coeff (1/s)", 0.0, 0.2, 0.05, step=0.01)

        st.divider()
        live_anim = st.checkbox("Live animation", value=False)
        fps = st.slider("Animation FPS", 2, 30, 10, step=1)

    params = RocketParams(
        dry_mass_kg=500.0,
        isp_s=300.0,
        max_thrust_N=25000.0,
        max_gimbal_deg=20.0,
        wind_amp_mps=wind_amp_mps,
        wind_freq_hz=wind_freq_hz,
        wind_drag_coeff_1_per_s=wind_drag_coeff,
    )

    # Keep tolerances fixed to preserve landing meaning.
    tolerances = LandingTolerances(
        x_error_m_abs=5.0,
        vx_abs_mps=6.0,
        vy_abs_mps=4.0,
        speed_abs_mps=12.0,
        require_descending=True,
    )

    if st.button("Run simulation", type="primary"):
        with st.spinner("Simulating descent..."):
            result = simulate_landing(
                params=params,
                target_x_m=target_x_m,
                initial_height_m=initial_height_m,
                initial_fuel_mass_kg=initial_fuel_mass_kg,
                dt_s=0.02,
                max_time_s=120.0,
                tolerances=tolerances,
            )

        if result.success:
            st.success("Landing successful.")
        else:
            st.error(f"Landing failed: {result.failure_reason}")

        # Metrics summary
        col1, col2, col3, col4 = st.columns(4)
        if result.touchdown_time_s is not None:
            col1.metric("Touchdown time", f"{result.touchdown_time_s:.2f} s")
            col2.metric("Velocity at touchdown", f"{result.touchdown_speed_mps:.2f} m/s")
            col3.metric("Fuel used", f"{result.touchdown_fuel_used_kg:.1f} kg")
            x_err = abs((result.touchdown_x_m or 0.0) - target_x_m)
            col4.metric("X error at touchdown", f"{x_err:.2f} m")
        else:
            col1.metric("Touchdown time", "N/A")
            col2.metric("Velocity at touchdown", "N/A")
            col3.metric("Fuel used", "N/A")
            col4.metric("X error at touchdown", "N/A")

        st.divider()

        history = result.history
        if not history or len(history.get("t_s", [])) == 0:
            st.write("No trajectory history to display.")
            return

        # Interactive visualization.
        n = len(history["t_s"])
        frame = st.slider("Animation frame", 0, n - 1, n - 1, step=1)

        if live_anim:
            placeholder = st.empty()
            # Cap number of rendered frames to keep Streamlit responsive.
            max_frames = min(250, n)
            for k in np.linspace(0, n - 1, max_frames, dtype=int):
                fig = _render_trajectory(history, target_x_m, frame=k)
                placeholder.pyplot(fig, clear_figure=True)
                time.sleep(1.0 / max(1, fps))
        else:
            fig = _render_trajectory(history, target_x_m, frame=frame)
            st.pyplot(fig, clear_figure=True)

        # Additional plot for dynamics.
        fig2 = _render_velocity(history)
        st.pyplot(fig2, clear_figure=True)

        # Optional: show a small debug table.
        with st.expander("Show raw metrics"):
            st.write({k: v for k, v in result.__dict__.items() if k != "history"})


if __name__ == "__main__":
    main()

