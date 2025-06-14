#!/usr/bin/env python3
# retraced.py ──────────────────────────────────────────────────────────
#  B&W and 3-layer colour film grain with optional supersampling
#  Grain parameters are now decoupled from output resolution.
# ----------------------------------------------------------------------
# 2025 © Ruitian Yang - MIT License
# ----------------------------------------------------------------------

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
cli.add_argument("--height", type=int, default=1080, help="Height of FINAL image (px)")

cli.add_argument(
    "--supersample",
    type=int,
    default=1,
    help="Oversampling factor (render at height*SS, then downscale)",
)

cli.add_argument(
    "--gamma", type=float, default=2.2, help="Electro-optical transfer γ (2.2 ≈ sRGB)"
)

# ❶  Grain parameters are given in *final* pixels
cli.add_argument(
    "--grain_radius",
    type=float,
    default=1.25,
    help="Mean grain radius, in final-image pixels",
)
cli.add_argument(
    "--grain_sigma",
    type=float,
    default=0.40,
    help="σ of log-normal radius, in final-image pixels",
)
cli.add_argument(
    "--sigma_filter", type=float, default=0.8, help="AA jitter σ, in final-image pixels"
)

cli.add_argument("--samples", type=int, default=100, help="# Monte-Carlo samples")

cli.add_argument(
    "--color", action="store_true", help="Enable three-layer colour negative (B→G→R)"
)

cli.add_argument("--output", default="out.png", help="Output filename")
args = cli.parse_args()

# ───────────────────── Derived sizes ──────────────────────────────────
SS = max(1, args.supersample)
H_final = args.height
H_sim = H_final * SS  # off-screen height

# ⟹ convert film parameters from final-px to simulation-px
R_px = args.grain_radius * SS
SIG_px = args.grain_sigma * SS
SIG_F = args.sigma_filter * SS

# ───────────────────── Init Taichi ────────────────────────────────────
ti.init(arch=ti.gpu, default_ip=ti.i32, random_seed=0)
print("[Taichi] arch:", ti.cfg.arch)


# ─────── Helper: load channels & resize to simulation size ────────────
def load_resized(im: Image.Image, height: int) -> np.ndarray:
    w = round(im.width * height / im.height)
    # --- sRGB → linear ------------------------------------------------
    srgb = np.asarray(im.resize((w, height), Image.LANCZOS), np.float32) * (1 / 255)
    g = args.gamma
    lin = np.where(srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, g))
    return lin.astype(np.float32)


pil_in = Image.open(args.input)

if args.color:
    pil_in = pil_in.convert("RGB")
    src_layers: List[np.ndarray] = [
        load_resized(pil_in.getchannel(ch), H_sim)  # B, G, R order
        for ch in ("B", "G", "R")
    ]
else:
    pil_in = pil_in.convert("L")
    src_layers = [load_resized(pil_in, H_sim)]

H, W = src_layers[0].shape
print(f"[Info] simulation size: {W}×{H}  (SS×{SS})")

# ──────────────── Taichi fields & physical constants ─────────────────
src_tex = ti.field(ti.f32, shape=(H, W))
neg = ti.field(ti.f32, shape=(H, W))

S = args.samples
R, SIG = R_px, SIG_px  # now *simulation* pixels
uMax, π = 1.0, math.pi
ag = 1.0 / math.ceil(1.0 / R)
R2 = R * R
sigma2_ln = 0.0 if SIG == 0 else math.log((SIG / R) ** 2 + 1)
sigma_ln = math.sqrt(sigma2_ln) if SIG > 0 else 0.0
mu_ln = math.log(R) - 0.5 * sigma2_ln
maxR = R if SIG == 0 else math.exp(mu_ln + 3.0902 * sigma_ln)
λ_fac = ag * ag / (π * (R2 + SIG * SIG))
eps = 1e-5
U32 = ti.u32


# ────────────────────────── RNG helpers ──────────────────────────────
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
    return Rnd(ns, ns * (1 / 4294967295.0))


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


# ────────────────────────── Render kernel ────────────────────────────
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
            ix = ti.cast(tm.clamp(ti.floor(xG), 0, W - 1), ti.i32)
            iy = ti.cast(tm.clamp(ti.floor(yG), 0, H - 1), ti.i32)

            u = tm.clamp(src_tex[iy, ix], 0.0, uMax - eps)
            lam = -λ_fac * ti.log(1.0 - u)

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


# ───────────────────── Render all layers ─────────────────────────────
results: List[np.ndarray] = []
for layer, chan in enumerate(src_layers):  # B → G → R
    src_tex.from_numpy(chan)
    print(f"[Layer {layer}] samples={S}")
    render(12345)  # shared RNG for neutral grain
    results.append(neg.to_numpy())


# ───────────────────── Down-scale helper ──────────────────────────────
def downscale(np_img: np.ndarray) -> Image.Image:
    """Resize from simulation to final resolution with Lanczos."""
    h = H_final
    w = round(np_img.shape[1] / SS)
    # return Image.fromarray((np_img * 255).astype(np.uint8)).resize(
    #     (w, h), Image.LANCZOS
    # )
    g = args.gamma
    srgb = np.where(
        np_img <= 0.0031308, np_img * 12.92, 1.055 * np.power(np_img, 1 / g) - 0.055
    )
    srgb = np.clip(srgb, 0, 1)
    return Image.fromarray((srgb * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)


# ───────────────────── Save final images ──────────────────────────────
root = Path(args.output).stem
ext = Path(args.output).suffix or ".png"

if args.color:
    neg_bgr = np.stack(results, axis=-1)
    neg_rgb = neg_bgr[..., ::-1]
    pos_rgb = 1.0 - neg_rgb
    downscale(pos_rgb).save(f"{root}{ext}")
else:
    pos_np = 1.0 - results[0]
    downscale(pos_np).save(f"{root}{ext}")

print("[OK] saved →", f"{root}{ext}")
