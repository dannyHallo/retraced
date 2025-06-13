#!/usr/bin/env python3
# ==============================================================
#  film_ray_min.py – minimal silver-halide toy ray tracer
# ==============================================================

import argparse, sys, os
from PIL import Image
import numpy as np
import taichi as ti

# --------------------------------------------------------------
#  CLI
# --------------------------------------------------------------
cli = argparse.ArgumentParser(description="Toy film simulator (minimal)")
cli.add_argument("--input", required=True, help="source image")
cli.add_argument("--height", type=int, default=2048, help="output vertical px")
cli.add_argument("--coverage", type=float, default=0.95, help="grain fill 0-1")
cli.add_argument("--alpha", type=float, default=2.0, help="Beer-Lambert α")
cli.add_argument("--rays", type=int, default=8, help="# MC rays")
cli.add_argument("--output", default="film_bw.png")
cli.add_argument(
    "--arch", default="auto", choices=["auto", "cpu", "cuda", "vulkan", "metal"]
)
args = cli.parse_args()


# --------------------------------------------------------------
#  Taichi init
# --------------------------------------------------------------
def pick(arch):
    if arch != "auto":
        return getattr(ti, arch)
    return ti.gpu if ti.is_gpu_supported() else ti.cpu


ti.init(arch=pick(args.arch), random_seed=0)
print("[Taichi] arch:", ti.cfg.arch)

# --------------------------------------------------------------
#  Load source image
# --------------------------------------------------------------
src_im = Image.open(args.input).convert("L")
H, W = args.height, round(src_im.width * args.height / src_im.height)
src_np = np.asarray(src_im.resize((W, H), Image.LANCZOS), np.float32) / 255.0

# --------------------------------------------------------------
#  Load blue-noise textures
#   – vec3 : single tile (unchanged)
#   – real : 64 tiles → 3-D stack
# --------------------------------------------------------------
BN = 1024

vec3_bn_im = Image.open("res/vec3.png").convert("L")
if vec3_bn_im.size != (BN, BN):
    sys.exit("vec3 blue-noise must be {BN}×{BN}")
vec3_bn_np = np.asarray(vec3_bn_im, np.float32) / 255.0

REAL_BN = 128
REAL_SLICES = 64
real_bn_stack = []
for k in range(REAL_SLICES):
    p = os.path.join("res", "real", f"out_{k}.png")
    if not os.path.isfile(p):
        sys.exit(f"missing blue-noise slice: {p}")
    im = Image.open(p).convert("L")
    if im.size != (REAL_BN, REAL_BN):
        sys.exit(f"blue-noise slice {p} must be {REAL_BN}×{REAL_BN}")
    real_bn_stack.append(np.asarray(im, np.float32) / 255.0)

# numpy shape (BN, BN, 64)
real_bn_np = np.stack(real_bn_stack, axis=2)

# --------------------------------------------------------------
#  Fields
# --------------------------------------------------------------
expo = ti.field(ti.f32, shape=(H, W))  # incoming exposure
grain = ti.field(ti.u8, shape=(H, W))  # 0/1 grain mask
latent = ti.field(ti.f32, shape=(H, W))  # energy stored per grain
film = ti.field(ti.f32, shape=(H, W))  # final transmissivity

vec3_bn_tex = ti.field(ti.f32, shape=(BN, BN))
real_bn_tex = ti.field(ti.f32, shape=(REAL_BN, REAL_BN, REAL_SLICES))

expo.from_numpy(src_np)
vec3_bn_tex.from_numpy(vec3_bn_np)
real_bn_tex.from_numpy(real_bn_np)

thr = max(0.0, min(args.coverage, 1.0))
N_RAYS = args.rays
FILM_Z = 1.0
one_u8 = ti.cast(1, ti.u8)


# --------------------------------------------------------------
#  Kernels
# --------------------------------------------------------------
@ti.kernel
def place_grains():
    for i, j in grain:
        grain[i, j] = one_u8 if vec3_bn_tex[i % BN, j % BN] < thr else 0


@ti.kernel
def raytrace():
    for py, px in expo:
        E_src = expo[py, px]
        if grain[py, px]:  # interact only if grain exists
            for r in range(N_RAYS):  # r = ray index
                bn_val = ti.random()  # this is better!
                # bn_val = real_bn_tex[py % REAL_BN, px % REAL_BN, r % REAL_SLICES]
                depth = bn_val * FILM_Z  # deterministic blue-noise depth
                deposited = E_src * ti.exp(-depth) / N_RAYS
                ti.atomic_add(latent[py, px], deposited)


@ti.kernel
def develop(alpha: ti.f32):
    for i, j in film:
        T = ti.exp(-alpha * latent[i, j]) if grain[i, j] else 1.0
        film[i, j] = T


# --------------------------------------------------------------
#  Pipeline
# --------------------------------------------------------------
print("[Info] simulating", f"{H}×{W}", "coverage=", thr, "#rays=", N_RAYS)
place_grains()
raytrace()
develop(args.alpha)

out = (film.to_numpy() * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(out, "L").save(args.output)
print("[OK] saved →", args.output)
