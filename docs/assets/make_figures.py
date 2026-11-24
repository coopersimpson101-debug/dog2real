#!/usr/bin/env python
"""Regenerate the README figures from real runs of this repo (MuJoCo offscreen rendering).

    pip install matplotlib pillow
    python docs/assets/make_figures.py sitl     # stand-up GIF + strip + banner
    python docs/assets/make_figures.py walk     # policies/walk_v0.pt walking at vx = 0.5 m/s
    python docs/assets/make_figures.py curve    # docs/assets/walk_v0_progress.csv

Outputs (in docs/assets/):
  sitl_standup.gif / sitl_standup_strip.png  sim-in-the-loop: the unchanged RobotRuntime +
                                             SafetyLayer driving MuJoCo, no policy (stand-up ramp)
  walk_v0.gif                                the committed policy (policies/walk_v0.pt), nominal
                                             physics, command vx = 0.5 m/s
  training_curve.png                         from walk_v0_progress.csv (every 5th iteration of the
                                             2500-iteration training run's progress.csv)
  banner.svg                                 README banner; the motif is the robot's real
                                             side-view geometry + base height from the SITL run
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ACCENT = "#FBBF24"
BG, PANEL, FG, MUTED, GRID = "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#30363d"
LEGS = ["FL", "FR", "RL", "RR"]


def _font(size):
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


CROP = (90, 80, 560, 350)  # robot region of a 640x360 render from the "track" camera


def _label(img: np.ndarray, text: str) -> Image.Image:
    im = Image.fromarray(img).crop(CROP)
    d = ImageDraw.Draw(im, "RGBA")
    f = _font(14)
    w = d.textlength(text, font=f)
    d.rectangle([6, 6, 18 + w, 28], fill=(13, 17, 23, 210))
    d.text((12, 9), text, fill=(251, 191, 36, 255), font=f)
    return im


def _save_gif(frames: list[Image.Image], path: Path, fps: float):
    # one shared palette, no dithering: keeps the checkered floor from bloating the file
    ref = frames[len(frames) // 2].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    pal = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0,
                optimize=True)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


# --------------------------------------------------------------------------- SITL --
def sitl(duration=5.0, w=640, h=360):
    from dog2real.config import load_yaml
    from dog2real.deploy.hal.sim_backend import SimIMUView, SimRobot
    from dog2real.deploy.runtime import RobotRuntime, SimClock
    from dog2real.deploy.safety import SafetyConfig, SafetyLayer

    dcfg = load_yaml("configs/deploy.yaml")
    robot = load_yaml(dcfg.get("robot", "robot.yaml"))
    ctrl = dcfg["control"]
    scfg = dcfg.get("sim", {})
    sim = SimRobot(robot, latency_s=float(scfg.get("latency_s", 0.004)),
                   start_pose=scfg.get("start_pose", "lying"))
    safety = SafetyLayer(SafetyConfig.from_dicts(robot, dcfg.get("safety")))
    period = float(ctrl["policy_dt"])
    rt = RobotRuntime(robot, dcfg, lambda obs: np.zeros(12), sim, SimIMUView(sim),
                      SimClock(period, sim), safety, [0.0, 0.0, 0.0])
    frames, heights, times = [], [], []
    rt.start()
    ticks = int(duration / period)
    while rt.tick_count < ticks and safety.ok:
        rt.tick()
        rt.clock.wait()
        heights.append(sim.base_height())
        times.append(rt.tick_count * period)
        if rt.tick_count % 3 == 0 or rt.tick_count == 1:
            frames.append((times[-1], heights[-1], sim.render(w, h)))
    # skeleton (x-z side view) of the final standing pose, from MuJoCo body positions
    import mujoco
    m, d = sim.model, sim.data
    pos = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)].copy()  # noqa: E731
    fid = lambda n: d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)].copy()  # noqa: E731
    skel = {leg: [pos(f"{leg}_hip"), pos(f"{leg}_calf"), fid(f"{leg}_foot_site")] for leg in LEGS}
    base = pos("base")
    fault = safety.fault
    rt.shutdown()
    print(f"SITL: final base height {heights[-1]:.3f} m after {times[-1]:.1f} s, "
          f"start {heights[0]:.3f} m, safety fault: {fault or 'none'}")

    gif = [_label(img, f"sim-in-the-loop, no policy  t={t:4.2f}s  base h={hb:.2f}m")
           for t, hb, img in frames]
    _save_gif(gif, HERE / "sitl_standup.gif", fps=1 / (3 * period))

    # strip of 5 frames
    pick = [0.04, 1.0, 2.0, 3.0, duration]
    tiles = []
    for tp in pick:
        k = min(range(len(frames)), key=lambda i: abs(frames[i][0] - tp))
        t, hb, img = frames[k]
        tiles.append(_label(img, f"t = {t:.1f} s   base h = {hb:.2f} m"))
    tw, th = tiles[0].size
    strip = Image.new("RGB", (tw * len(tiles) + 6 * (len(tiles) - 1), th), BG)
    for i, tile in enumerate(tiles):
        strip.paste(tile, (i * (tw + 6), 0))
    strip.save(HERE / "sitl_standup_strip.png", optimize=True)
    print("wrote", HERE / "sitl_standup_strip.png")
    banner(skel, base, robot, np.array(times), np.array(heights))


# ------------------------------------------------------------------------- policy --
def walk(checkpoint="policies/walk_v0.pt", seconds=6.0, vx=0.5, w=640, h=360):
    import torch

    from dog2real.deploy.export import load_deploy_policy
    from dog2real.sim.env import VecQuadrupedEnv

    pol, cfg = load_deploy_policy(checkpoint)
    cfg["domain_rand"]["enabled"] = False
    cfg["imu_noise"]["enabled"] = False
    cfg["obs_noise"]["enabled"] = False
    env = VecQuadrupedEnv(cfg, num_envs=1, seed=0)
    obs, _ = env.reset()
    frames, falls, err = [], 0, []
    for k in range(int(seconds / env.dt)):
        env.commands[:] = [vx, 0.0, 0.0]
        obs, _ = env.compute_observations()
        with torch.no_grad():
            act = pol(torch.as_tensor(obs)).numpy()
        obs, _, _, info = env.step(act)
        falls += int(info["terminated"][0])
        s = env._state
        err.append(float(np.linalg.norm(s["commands"][0, :2] - s["base_lin_vel"][0, :2])))
        if k % 3 == 0:
            frames.append(_label(env.render(0, w, h),
                                 f"SIMULATION  walk_v0  cmd vx={vx} m/s  t={(k + 1) * env.dt:4.2f}s"))
    env.close()
    print(f"walk: cmd vx={vx}, mean |v_cmd - v| {np.mean(err):.3f} m/s, falls {falls} in {seconds:.0f} s")
    _save_gif(frames, HERE / "walk_v0.gif", fps=1 / (3 * env.dt))


# -------------------------------------------------------------------------- curve --
def curve(csv_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    keys = rows[0].keys()
    print("csv columns:", ", ".join(keys))
    col = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows])  # noqa: E731
    it = col("iteration")
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": FG, "text.color": FG,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
        "font.family": "monospace", "font.size": 10, "legend.facecolor": PANEL,
        "legend.edgecolor": GRID})
    panels = [(k, t) for k, t in [("train/mean_episode_length", "mean episode length [steps, max 1000]"),
                                  ("episode/rew_tracking_lin_vel", "lin-vel tracking reward (max 1.0)"),
                                  ("episode/cmd_vx_max", "curriculum: max commanded vx [m/s]")] if k in keys]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.2), dpi=140)
    for ax, (k, title) in zip(np.atleast_1d(axs), panels):
        y = col(k)
        ok = ~np.isnan(y)
        ax.plot(it[ok], y[ok], color=ACCENT, lw=0.8, alpha=0.35)
        win = 10  # rolling mean over 10 logged points (= 50 iterations)
        sm = np.convolve(y[ok], np.ones(win) / win, mode="valid")
        ax.plot(it[ok][win - 1:], sm, color=ACCENT, lw=1.8)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("PPO iteration")
        ax.grid(True, lw=0.4)
    fig.suptitle("walk_v0 training (simulation): 128 envs x 24 steps x 2500 PPO iterations = 7.7 M env steps, "
                 "77 min on a desktop CPU", x=0.01, ha="left", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(HERE / "training_curve.png")
    print("wrote", HERE / "training_curve.png")


# ------------------------------------------------------------------------- banner --
def banner(skel, base, robot, t, hgt):
    W, H = 1280, 400
    # side view: x right, z up. Scale the real body geometry into the right-hand box.
    s, ox, oz = 600.0, 1000.0, 300.0
    X = lambda p: ox + (p[0] - base[0]) * s  # noqa: E731
    Z = lambda p: oz - p[2] * s  # noqa: E731
    bl, _, bh = robot["base"]["size"]
    body = (f'<rect x="{X(base) - bl / 2 * s:.1f}" y="{Z(base) - bh / 2 * s:.1f}" width="{bl * s:.1f}" '
            f'height="{bh * s:.1f}" rx="10" fill="#161b22" stroke="#FBBF24" stroke-width="3"/>')
    legs = []
    for leg, (hip, knee, foot) in skel.items():
        far = leg in ("FR", "RR")
        op = 0.35 if far else 1.0
        if far:  # shift the far-side legs a little for depth
            hip, knee, foot = (p + np.array([0.025, 0.0, 0.012]) for p in (hip, knee, foot))
        pts = f"{X(hip):.1f},{Z(hip):.1f} {X(knee):.1f},{Z(knee):.1f} {X(foot):.1f},{Z(foot):.1f}"
        legs.append(f'<polyline points="{pts}" fill="none" stroke="#FBBF24" stroke-opacity="{op}" '
                    f'stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>'
                    f'<circle cx="{X(knee):.1f}" cy="{Z(knee):.1f}" r="6" fill="#0d1117" stroke="#FBBF24" '
                    f'stroke-opacity="{op}" stroke-width="2.5"/>'
                    f'<circle cx="{X(foot):.1f}" cy="{Z(foot):.1f}" r="{robot["leg"]["foot"]["radius"] * s:.1f}" '
                    f'fill="#FBBF24" fill-opacity="{op}"/>')
    far_legs = "".join(l for l, k in zip(legs, skel) if k in ("FR", "RR"))
    near_legs = "".join(l for l, k in zip(legs, skel) if k not in ("FR", "RR"))
    # base-height trace of the stand-up (real SITL data) along the floor line
    tx0, tx1, ty0 = 760, 1240, 372
    hs = (hgt - hgt.min()) / max(hgt.max() - hgt.min(), 1e-6)
    trace = " ".join(f"{tx0 + (tx1 - tx0) * ti / t[-1]:.1f},{ty0 - 34 * hi:.1f}"
                     for ti, hi in zip(t[::2], hs[::2]))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <pattern id="g" width="32" height="32" patternUnits="userSpaceOnUse">
      <path d="M32 0H0V32" fill="none" stroke="#e6edf3" stroke-opacity="0.05" stroke-width="1"/>
    </pattern>
    <radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#FBBF24" stop-opacity="0.14"/>
      <stop offset="1" stop-color="#FBBF24" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="{W}" height="{H}" fill="#0d1117"/>
  <rect width="{W}" height="{H}" fill="url(#g)"/>
  <ellipse cx="1000" cy="200" rx="300" ry="190" fill="url(#glow)"/>
  <line x1="740" y1="{oz:.0f}" x2="1260" y2="{oz:.0f}" stroke="#30363d" stroke-width="2"/>
  {far_legs}
  {body}
  {near_legs}
  <polyline points="{trace}" fill="none" stroke="#FBBF24" stroke-opacity="0.8" stroke-width="2"/>
  <text x="{tx1}" y="{ty0 + 20}" text-anchor="end" fill="#8b949e" font-size="13"
        font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">MuJoCo pose + base height, sim-in-the-loop stand-up</text>
  <g font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">
    <rect x="80" y="118" width="6" height="96" fill="#FBBF24"/>
    <text x="108" y="190" fill="#e6edf3" font-size="80" font-weight="700">dog2real</text>
    <text x="110" y="248" fill="#8b949e" font-size="22">Sim-to-real RL for a DIY 12-DoF robot dog</text>
    <text x="110" y="280" fill="#8b949e" font-size="22">MuJoCo · from-scratch PPO · ONNX · MIT-mode CAN</text>
    <text x="110" y="330" fill="#FBBF24" font-size="15">one YAML robot → train → randomise → export → deploy</text>
  </g>
</svg>
'''
    (HERE / "banner.svg").write_text(svg, encoding="utf-8")
    print("wrote", HERE / "banner.svg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["sitl", "walk", "curve"])
    ap.add_argument("--checkpoint", default="policies/walk_v0.pt")
    ap.add_argument("--csv", default=str(HERE / "walk_v0_progress.csv"))
    a = ap.parse_args()
    if a.what == "sitl":
        sitl()
    elif a.what == "walk":
        walk(a.checkpoint)
    else:
        curve(a.csv)
