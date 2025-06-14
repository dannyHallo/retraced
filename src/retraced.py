#!/usr/bin/env python3
# retraced.py ──────────────────────────────────────────────────────────
#  B&W + 3-layer colour film grain with optional supersampling
#  © 2025 Ruitian Yang  - MIT License
# ---------------------------------------------------------------------
import math, argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import taichi as ti
import taichi.math as tm

# ───────────────────────── CLI ────────────────────────────────────────
cli = argparse.ArgumentParser(description="Physically-based film-grain emulator")
cli.add_argument("--input", required=True, help="Input image")

cli.add_argument("--height", type=int, default=1080, help="Final image height")
cli.add_argument("--supersample", type=int, default=1, help="Supersampling factor")
cli.add_argument(
    "--grain_radius", type=float, default=1.25, help="Mean grain radius (px)"
)
cli.add_argument(
    "--grain_sigma", type=float, default=0.40, help="σ of log-normal radius"
)
cli.add_argument("--sigma_filter", type=float, default=0.8, help="AA jitter σ (px)")
cli.add_argument("--lambda_scale", type=float, default=3.0, help="Exposure scaling")
cli.add_argument("--samples", type=int, default=100, help="# Monte-Carlo samples")
cli.add_argument(
    "--color", action="store_true", help="Enable three-layer colour negative (B→G→R)"
)
cli.add_argument("--output", default="out.png", help="Output filename")
args = cli.parse_args()

# Derived sizes --------------------------------------------------------
SS = max(1, args.supersample)
H_final = args.height
H_sim = H_final * SS  # simulation (oversample) height

# ─────────────────────── Init Taichi ──────────────────────────────────
ti.init(arch=ti.gpu, default_ip=ti.i32, random_seed=0)
print("[Taichi] arch:", ti.cfg.arch)


# ──────────────────── Helper: prepare sources ─────────────────────────
def load_and_resize(im: Image.Image, height: int) -> np.ndarray:
    h = height
    w = round(im.width * h / im.height)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) * (1 / 255)


pil_in = Image.open(args.input)

if args.color:
    pil_in = pil_in.convert("RGB")
    src_channels: List[np.ndarray] = [
        load_and_resize(pil_in.getchannel(ch), H_sim)  # B, G, R order
        for ch in ("B", "G", "R")
    ]
else:
    pil_in = pil_in.convert("L")
    src_channels = [load_and_resize(pil_in, H_sim)]

H, W = src_channels[0].shape
print(f"[Info] simulation size {W}×{H}  (SS×{SS})")

# ───────────────────────── Fields & consts ────────────────────────────
src_tex = ti.field(ti.f32, shape=(H, W))
neg = ti.field(ti.f32, shape=(H, W))

S = args.samples
R, SIG = args.grain_radius, args.grain_sigma
SIG_F = args.sigma_filter
uMax, π = 2.0, math.pi
ag = 1.0 / math.ceil(1.0 / R)
R2 = R * R
sigma2_ln = 0.0 if SIG == 0 else math.log((SIG / R) ** 2 + 1)
sigma_ln = math.sqrt(sigma2_ln) if SIG > 0 else 0.0
mu_ln = math.log(R) - 0.5 * sigma2_ln
maxR = R if SIG == 0 else math.exp(mu_ln + 3.0902 * sigma_ln)
λ_fac = ag * ag / (π * (R2 + SIG * SIG))
λ_scale = args.lambda_scale
eps = 1e-5
U32 = ti.u32


# ───────────────────────── RNG helpers ────────────────────────────────
@ti.func
def wang(seed: U32) -> U32:
    seed = (seed ^ 61) ^ (seed >> 16)
    seed *= 9
    seed ^= seed >> 4
    seed *= 668265261
    seed ^= seed >> 15
    return seed


@ti.func
def xor_shift(s: U32) -> U32:
    s ^= s << 13
    s ^= s >> 17
    s ^= s << 5
    return s


@ti.dataclass
class Rnd:
    s: U32
    v: ti.f32


@ti.func
def rnd01(state: U32) -> Rnd:
    ns = xor_shift(state)
    return Rnd(ns, ns * (1.0 / 4294967295.0))


@ti.func
def rnd_gauss(state: U32) -> Rnd:
    r1 = rnd01(state)
    r2 = rnd01(r1.s)
    r = ti.sqrt(-2 * ti.log(r1.v + 1e-12))
    return Rnd(r2.s, r * ti.cos(2 * π * r2.v))


@ti.func
def rnd_poisson(state: U32, lam, expLam) -> Rnd:
    r = rnd01(state)
    u = r.v
    x, prod, summ = U32(0), expLam, expLam
    lim = ti.cast(ti.floor(1e4 * lam), U32)
    while (u > summ) and (x < lim):
        x += 1
        prod *= lam / ti.cast(x, ti.f32)
        summ += prod
    return Rnd(r.s, x)


@ti.func
def sq(a, b, c, d):
    return (a - c) * (a - c) + (b - d) * (b - d)


# ─────────────────────────── Kernel ───────────────────────────────────
@ti.kernel
def render(seed: U32):
    fseed = wang(seed)
    for py, px in neg:
        st = wang(U32(py * 73856093 ^ px * 19349663 ^ fseed))
        hit = 0.0
        for _ in range(S):
            g1 = rnd_gauss(st)
            st = g1.s
            g2 = rnd_gauss(st)
            st = g2.s
            xG = ti.cast(px, ti.f32) + SIG_F * g1.v
            yG = ti.cast(py, ti.f32) + SIG_F * g2.v
            ix = ti.cast(tm.clamp(ti.floor(xG), 0.0, W - 1.0), ti.i32)
            iy = ti.cast(tm.clamp(ti.floor(yG), 0.0, H - 1.0), ti.i32)

            u = src_tex[iy, ix] / (uMax + eps)
            lam = λ_scale * (-λ_fac * ti.log(1.0 - u))
            exL = ti.exp(-lam)

            minX = ti.cast(ti.floor((xG - maxR) / ag), ti.i32)
            maxX = ti.cast(ti.floor((xG + maxR) / ag), ti.i32)
            minY = ti.cast(ti.floor((yG - maxR) / ag), ti.i32)
            maxY = ti.cast(ti.floor((yG + maxR) / ag), ti.i32)

            covered = False
            cx = minX
            while cx <= maxX and not covered:
                cy = minY
                while cy <= maxY and not covered:
                    cs = wang(U32((cy & 0xFFFF) << 16 | (cx & 0xFFFF)) + fseed)
                    rP = rnd_poisson(cs, lam, exL)
                    cs = rP.s
                    # don't use underscore here, because it will collide with the underscore defined before, said by Taichi
                    for z in range(ti.cast(rP.v, ti.i32)):
                        ru = rnd01(cs)
                        cs = ru.s
                        rv = rnd01(cs)
                        cs = rv.s
                        xc = ag * (ti.cast(cx, ti.f32) + ru.v)
                        yc = ag * (ti.cast(cy, ti.f32) + rv.v)
                        r2 = R2
                        if SIG > 0:
                            rg = rnd_gauss(cs)
                            cs = rg.s
                            rad = ti.min(ti.exp(mu_ln + sigma_ln * rg.v), maxR)
                            r2 = rad * rad
                        if sq(xc, yc, xG, yG) < r2:
                            covered = True
                            break
                    cy += 1
                cx += 1
            hit += 1 if covered else 0
        neg[py, px] = 1.0 - hit / S


# ───────────────────────── Simulation ────────────────────────────────
results: List[np.ndarray] = []
for layer, chan_np in enumerate(src_channels):  # B → G → R
    src_tex.from_numpy(chan_np)
    print(f"[Layer {layer}] samples={S}")
    render(12345)
    results.append(neg.to_numpy())  # pull to CPU

# ───────────────────────── Down-sampling ─────────────────────────────
root = Path(args.output).stem
ext = Path(args.output).suffix or ".png"


def downscale(img_np: np.ndarray) -> Image.Image:
    """Lanczos resize from simulation to final size."""
    h_out = H_final
    w_out = round(img_np.shape[1] / SS)
    pil = Image.fromarray((img_np * 255).astype(np.uint8))
    return pil.resize((w_out, h_out), Image.LANCZOS)


if args.color:
    neg_bgr = np.stack(results, axis=-1)  # B, G, R
    neg_rgb = neg_bgr[..., ::-1]  # R, G, B
    pos_rgb = 1.0 - neg_rgb

    downscale(pos_rgb).save(f"{root}{ext}")
else:
    neg_np = results[0]
    pos_np = 1.0 - neg_np
    downscale(pos_np).save(f"{root}{ext}")

print("[OK] saved →", f"{root}{ext}")
