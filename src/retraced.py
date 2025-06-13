#!/usr/bin/env python3
# ==============================================================
#  film_ray.py – toy silver-halide ray tracer in Taichi
# ==============================================================

import os, sys, argparse, traceback, faulthandler
from PIL import Image
import numpy as np
import taichi as ti

# ------------------------------------------------------------------
#  DEBUG / OOM-MESSAGE HELPER
# ------------------------------------------------------------------
faulthandler.enable(all_threads=True)


def _excepthook(exc_type, exc, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        return sys.__excepthook__(exc_type, exc, tb)
    traceback.print_exception(exc_type, exc, tb)
    sys.exit(1)


sys.excepthook = _excepthook


def safe(kernel, *a, **kw):
    """Launch `kernel` and sync so allocation errors arise *here*."""
    kernel(*a, **kw)
    ti.sync()


# ------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------
cli = argparse.ArgumentParser(
    description="Fast blue-noise film demo " "with Monte-Carlo ray tracing"
)
cli.add_argument("--input", required=True, help="source image")
cli.add_argument("--bn", default="res/bn.png", help="1024² blue-noise LUT")
cli.add_argument(
    "-H",
    "--height",
    type=int,
    default=2048,
    help="vertical resolution of the scanned negative",
)
cli.add_argument(
    "--scale", type=int, default=4, help="super-sampling factor inside the film"
)
cli.add_argument(
    "--coverage",
    type=float,
    default=0.95,
    help="fraction of area covered by grains (0–1)",
)
cli.add_argument(
    "--alpha", type=float, default=2.0, help="contrast of the negative (Beer–Lambert α)"
)
cli.add_argument(
    "--rays", type=int, default=8, help="number of rays fired per source pixel"
)
cli.add_argument("--output", default="film_bw.png")
cli.add_argument(
    "--arch", default="auto", choices=["auto", "cpu", "cuda", "vulkan", "metal"]
)
args = cli.parse_args()


# ------------------------------------------------------------------
#  Taichi init
# ------------------------------------------------------------------
def pick(a: str):
    if a != "auto":
        return getattr(ti, a)
    try:
        return ti.gpu
    except ti.TaichiRuntimeError:
        return ti.cpu


os.environ["TI_DEVICE_MEMORY_GB"] = "4"  # limit, so OOM → Python
ti.init(arch=pick(args.arch), debug=True, log_level=ti.DEBUG, random_seed=0)

print("[Taichi] arch =", ti.lang.impl.current_cfg().arch)

# ------------------------------------------------------------------
#  Load input bitmap and blue-noise LUT
# ------------------------------------------------------------------
im = Image.open(args.input).convert("L")
H_scan = args.height
W_scan = round(im.width * H_scan / im.height)
im = im.resize((W_scan, H_scan), Image.LANCZOS)
src_np = np.asarray(im, np.float32) / 255.0  # 0–1

lut = Image.open(args.bn).convert("L")
if lut.size != (1024, 1024):
    print("blue-noise texture must be 1024×1024", file=sys.stderr)
    sys.exit(1)
bn_np = np.asarray(lut, np.uint8).astype(np.float32) / 255.0
BN = 1024

# ------------------------------------------------------------------
#  Canvas sizes
# ------------------------------------------------------------------
S = args.scale
H_hi, W_hi = H_scan * S, W_scan * S
print(f"[Info] scan {H_scan}×{W_scan}   canvas {H_hi}×{W_hi}   " f"scale {S}")

# ------------------------------------------------------------------
#  Taichi fields
# ------------------------------------------------------------------
expo_lo = ti.field(ti.f32, shape=(H_scan, W_scan))
expo_lo.from_numpy(src_np)

expo_hi = ti.field(ti.f32, shape=(H_hi, W_hi))  # up-sampled intensity
grain = ti.field(ti.u8, shape=(H_hi, W_hi))  # 0/1 grain presence
latent = ti.field(ti.f32, shape=(H_hi, W_hi))  # exposure per grain
bn_tex = ti.field(ti.f32, shape=(BN, BN))
bn_tex.from_numpy(bn_np)

film = ti.field(ti.f32, shape=(H_scan, W_scan))  # final negative


# ------------------------------------------------------------------
#  Kernels
# ------------------------------------------------------------------
@ti.kernel
def upsample():
    for i, j in expo_hi:
        expo_hi[i, j] = expo_lo[i // S, j // S]


thr = max(0.0, min(args.coverage, 0.9999))
print(f"[Info] effective coverage threshold = {thr:.4f}")

one_u8 = ti.cast(1, ti.u8)
zero_u8 = ti.cast(0, ti.u8)


@ti.kernel
def place_grains():
    for i, j in grain:
        grain[i, j] = one_u8 if bn_tex[i % BN, j % BN] < thr else zero_u8


N_RAYS = args.rays
FILM_Z = 1.0  # thickness in arbitrary units
G_RADIUS = 0.5  # each grain occupies exactly one high-res pixel


@ti.kernel
def raytrace():
    ti.loop_config(serialize=True)  # more stable on GPU
    for py, px in expo_lo:  # low-res pixel loop
        src_E = expo_lo[py, px]
        base_i, base_j = py * S, px * S
        for s in range(N_RAYS):
            # --- random sub-pixel launch position -----------------
            uf, vf = ti.random(ti.f32), ti.random(ti.f32)
            iy = base_i + ti.cast(uf * S, ti.i32)
            ix = base_j + ti.cast(vf * S, ti.i32)

            # --- ray direction: straight into film (0,0,1) --------
            # Here we only care about z, because grains are flat
            # Choose a random depth at which the ray first *could* meet a grain
            depth = ti.random(ti.f32) * FILM_Z

            # Only interact if a grain is present
            if grain[iy, ix]:
                # simple Lambert–Beer: energy decays with depth
                E_arriving = src_E * ti.exp(-depth)
                ti.atomic_add(latent[iy, ix], E_arriving / N_RAYS)
                # ray absorbed, stop
            # else: no grain at that column, ray exits film


@ti.kernel
def develop(alpha: ti.f32):
    for i, j in latent:
        T = 1.0  # defined unconditionally
        if grain[i, j]:
            T = ti.exp(-alpha * latent[i, j])
        expo_hi[i, j] = T


@ti.kernel
def downsample():
    for py, px in film:
        acc = 0.0
        for di, dj in ti.ndrange(S, S):
            acc += expo_hi[py * S + di, px * S + dj]
        film[py, px] = acc / (S * S)


# ------------------------------------------------------------------
#  Pipeline
# ------------------------------------------------------------------
print("[Info] processing", args.input)
safe(upsample)
safe(place_grains)
safe(raytrace)
safe(develop, args.alpha)
safe(downsample)

out = (film.to_numpy() * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(out, "L").save(args.output)
print("[OK] saved →", args.output)
