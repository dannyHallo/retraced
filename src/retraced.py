#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colour- & B/W-negative film-grain emulator
"""
# ───────────────────────── imports ───────────────────────────────────────────
import argparse, math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import taichi as ti, taichi.math as tm
import tomllib as toml

# ───────────────────────── CLI ───────────────────────────────────────────────
cli = argparse.ArgumentParser(description="Colour-negative film-grain emulator")
cli.add_argument("--input", required=True)
cli.add_argument("--height", type=int, default=1080)
cli.add_argument("--supersample", type=int, default=1)
cli.add_argument("--samples", type=int, default=200)
cli.add_argument("--bounce_samples", type=int, default=200)
cli.add_argument("--gamma", type=float, default=2.2)
cli.add_argument("--film_cfg", default="film-config.toml")
cli.add_argument("--output", default="out.png")
args = cli.parse_args()

# ── read TOML while preserving block order (for readable logs) ───────────────
cfg_txt = Path(args.film_cfg).read_text(encoding="utf8")
order: List[Tuple[str, int]] = []
_seen = {"emulsion": 0, "filter": 0, "film_base": 0, "back": 0}
for ln in cfg_txt.splitlines():
    ln = ln.strip()
    for k in _seen:
        if ln.startswith(f"[[{k}]]"):
            order.append((k, _seen[k]))
            _seen[k] += 1

cfg = toml.loads(cfg_txt)
emulsions = cfg.get("emulsion", [])
filters = cfg.get("filter", [])
bases = cfg.get("film_base", [])
backs = cfg.get("back", [])

if not emulsions:
    raise ValueError("Need at least one [[emulsion]] block")
if len(filters) not in {len(emulsions), len(emulsions) - 1}:
    raise ValueError("#[[filter]] must equal #[[emulsion]] or be one fewer")

if not bases:
    bases.append({"thickness": 1.0})
if not backs:
    backs.append({"reflectance": 0.0})

film_thick = float(bases[0].get("thickness", 1.0))
back_refl = float(backs[0].get("reflectance", 0.0))
EPS = 1e-5


# ─────────────── colour‐space helpers ────────────────────────────────────────
def srgb_to_linear(a, g):
    a = a / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** g).astype(
        np.float32
    )


def linear_to_srgb(a, g):
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / g) - 0.055)


# ───────────────────── Taichi + input image ──────────────────────────────────
ti.init(arch=ti.gpu, default_ip=ti.i32, random_seed=0)

pil_in = Image.open(args.input).convert("RGB")
H_sim = args.height * args.supersample
W_sim = round(pil_in.width * H_sim / pil_in.height)
img_lin = srgb_to_linear(
    np.asarray(pil_in.resize((W_sim, H_sim), Image.LANCZOS), np.float32), args.gamma
)

# ─────────────── Taichi fields for grain simulation ──────────────────────────
src_tex = ti.field(ti.f32, shape=(H_sim, W_sim))  # scene luminance for exposure
neg = ti.field(ti.f32, shape=(H_sim, W_sim))  # 1→clear, 0→opaque

R_f, SIG_f, SIGF_f, R2_f = (ti.field(ti.f32, shape=()) for _ in range(4))
sigma_ln_f, mu_ln_f, maxR_f = (ti.field(ti.f32, shape=()) for _ in range(3))
lambda_fac_f, ag_f = (ti.field(ti.f32, shape=()) for _ in range(2))


# ────────────────────── RNG helpers (Wang & XOR-shift) ───────────────────────
@ti.func
def wang(seed: ti.u32) -> ti.u32:
    seed = (seed ^ 61) ^ (seed >> 16)
    seed *= 9
    seed ^= seed >> 4
    seed *= 668265261
    seed ^= seed >> 15
    return seed


@ti.func
def xor_shift(s: ti.u32) -> ti.u32:
    s ^= s << 13
    s ^= s >> 17
    s ^= s << 5
    return s


@ti.dataclass
class Rnd:
    s: ti.u32
    v: ti.f32


@ti.func
def rnd01(state):
    ns = xor_shift(state)
    return Rnd(ns, ns * (1.0 / 4294967295.0))


@ti.func
def rnd_gauss(state):
    r1 = rnd01(state)
    r2 = rnd01(r1.s)
    r = ti.sqrt(-2.0 * ti.log(r1.v + 1e-12))
    return Rnd(r2.s, r * ti.cos(2 * math.pi * r2.v))


@ti.func
def rnd_poisson(state, lam, expLam):
    r = rnd01(state)
    u = r.v
    x = ti.u32(0)
    prod = summ = expLam
    lim = ti.cast(ti.floor(1e4 * lam), ti.u32)
    while (u > summ) and (x < lim):
        x += 1
        prod *= lam / ti.cast(x, ti.f32)
        summ += prod
    return Rnd(r.s, ti.cast(x, ti.f32))


@ti.func
def sq(a, b, c, d):
    return (a - c) * (a - c) + (b - d) * (b - d)


# ───────────────────────── grain kernel ──────────────────────────────────────
@ti.kernel
def render(seed: ti.u32, n_samples: ti.i32):
    fseed = wang(seed)
    SIG, SIG_F, R2 = SIG_f[None], SIGF_f[None], R2_f[None]
    sigma_ln, mu_ln = sigma_ln_f[None], mu_ln_f[None]
    maxR, λ_fac, ag = maxR_f[None], lambda_fac_f[None], ag_f[None]

    for py, px in neg:
        st = wang(ti.u32(py * 73856093 ^ px * 19349663 ^ fseed))
        hit = 0.0
        for _ in range(n_samples):
            g1 = rnd_gauss(st)
            st = g1.s
            g2 = rnd_gauss(st)
            st = g2.s
            xG = ti.cast(px, ti.f32) + SIG_F * g1.v
            yG = ti.cast(py, ti.f32) + SIG_F * g2.v
            ix = ti.cast(tm.clamp(ti.floor(xG), 0, W_sim - 1), ti.i32)
            iy = ti.cast(tm.clamp(ti.floor(yG), 0, H_sim - 1), ti.i32)

            u = tm.clamp(src_tex[iy, ix], 0.0, 1.0 - EPS)
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
                    cs = wang(ti.u32(((cy & 0xFFFF) << 16) | (cx & 0xFFFF)) + fseed)
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
            hit += 1.0 if covered else 0.0
        neg[py, px] = 1.0 - hit / n_samples  # 1→clear, 0→opaque


# ─── physics parameter packing helper ────────────────────────────────────────
def prep_physics(r_px, sig_px, sigF_px):
    R_f[None], SIG_f[None], SIGF_f[None] = r_px, sig_px, sigF_px
    R2_f[None] = r_px * r_px
    if sig_px == 0:
        sigma2_ln = sigma_ln = 0.0
    else:
        sigma2_ln = math.log((sig_px / r_px) ** 2 + 1)
        sigma_ln = math.sqrt(sigma2_ln)
    sigma_ln_f[None] = sigma_ln
    mu_ln_f[None] = math.log(r_px) - 0.5 * sigma2_ln
    maxR_f[None] = r_px if sig_px == 0 else math.exp(mu_ln_f[None] + 3.0902 * sigma_ln)
    ag = 1.0 / math.ceil(1.0 / r_px) if r_px > 0 else 1.0
    ag_f[None] = ag
    lambda_fac_f[None] = (
        ag * ag / (math.pi * (r_px * r_px + sig_px * sig_px)) if r_px > 0 else 0.0
    )


# ─── helper: build (absorb_strength, exposure_weights) from dye colour ───────
def build_vectors(dye_rgb: List[int]):
    dye = np.asarray(dye_rgb, np.float32) / 255.0
    comp = 1.0 - dye  # what the developed silver blocks
    # Panchromatic B/W layer
    if np.allclose(comp, 1.0):
        absorb = np.array([1.0, 1.0, 1.0], np.float32)  # can block every colour fully
        exposeW = np.array(
            [1 / 3, 1 / 3, 1 / 3], np.float32
        )  # uniform spectral sensitivity
        return absorb, exposeW
    # one-hot layer (colour-negative)
    if np.count_nonzero(comp) == 1:
        return comp.astype(np.float32), comp.astype(np.float32)
    # general chromatic mixture
    s = comp.sum()
    exposeW = comp / s if s > 0 else comp
    return comp.astype(np.float32), exposeW.astype(np.float32)


# ─────────────────────────── main traversal ─────────────────────────────────
ray_front = img_lin.copy()  # what the next emulsion “sees”
layer_absorbs = []  # absorb_strength vectors
layer_dens = []  # neg α (0–1)
layer_depths = []

current_z = 0.0
print()
for idx, (kind, k) in enumerate(order):
    if kind == "emulsion":
        emu = emulsions[k]
        absorb, expoW = build_vectors(emu["sensitising_dye_color"])

        # exposure map (single channel, 0→max, 1→min grain density)
        src = 1.0 - np.clip(np.tensordot(ray_front, expoW, axes=([-1], [0])), 0.0, 1.0)
        src_tex.from_numpy(src.astype(np.float32))

        r_px = emu.get("grain_radius", 0.02) * args.supersample
        sig_px = emu.get("grain_sigma", 0.00) * args.supersample
        sig_f = emu.get("sigma_filter", 0.08) * args.supersample
        prep_physics(r_px, sig_px, sig_f)

        print(
            f"[emu {idx}] dye={emu['sensitising_dye_color']}  R={r_px/args.supersample:.3f}px"
        )
        render(12345 + idx * 19, args.samples)
        dens = neg.to_numpy()

        layer_absorbs.append(absorb)
        layer_dens.append(dens)
        layer_depths.append(current_z)
        current_z += 1.0
        ray_front *= 1.0 - absorb[None, None, :] * (1.0 - dens[..., None])

    elif kind == "filter":
        col = np.asarray(filters[k]["color"], np.float32) / 255.0
        ray_front *= col[None, None, :]
        print(f"[filter {idx}] colour={filters[k]['color']}")
    elif kind == "film_base":
        print(f"[base] thickness={film_thick}")
    elif kind == "back":
        print(f"[back] reflectance={back_refl}")

# ───────────────────────── forward transmittance ────────────────────────────
front_rgb = np.ones_like(img_lin)
for absorb, dens in zip(layer_absorbs, layer_dens):
    front_rgb *= 1.0 - absorb[None, None, :] + absorb[None, None, :] * dens[..., None]

# ───────────────────── back-bounce / halation (optional) ─────────────────────
if back_refl > EPS:
    n_layers = len(layer_dens)
    front_rgb_f = ti.Vector.field(3, ti.f32, shape=(H_sim, W_sim))
    bounced_rgb_f = ti.Vector.field(3, ti.f32, shape=(H_sim, W_sim))
    dens_f = ti.field(ti.f32, shape=(n_layers, H_sim, W_sim))
    absorb_f = ti.Vector.field(3, ti.f32, shape=n_layers)
    depth_f = ti.field(ti.f32, shape=n_layers)

    front_rgb_f.from_numpy(front_rgb.astype(np.float32))
    dens_f.from_numpy(np.stack(layer_dens, 0).astype(np.float32))
    absorb_f.from_numpy(np.stack(layer_absorbs, 0).astype(np.float32))
    depth_f.from_numpy(
        np.asarray(layer_depths, np.float32) / max(layer_depths[-1], 1.0) * film_thick
    )

    S, film_px = args.bounce_samples, film_thick

    @ti.func
    def hemi_cosine(st):
        r1 = rnd01(st)
        r2 = rnd01(r1.s)
        z = ti.sqrt(r1.v)
        r = ti.sqrt(1.0 - z * z)
        phi = 2 * math.pi * r2.v
        return Rnd(r2.s, tm.vec3(r * ti.cos(phi), r * ti.sin(phi), z))

    @ti.kernel
    def bounce(seed: ti.u32):
        fseed = wang(seed)
        for py, px in bounced_rgb_f:
            st = wang(ti.u32(py * 9781 ^ px * 6271 ^ fseed))
            acc = tm.vec3(0.0)
            x0, y0 = ti.cast(px, ti.f32), ti.cast(py, ti.f32)
            for _ in range(S):
                dir = hemi_cosine(st)
                st = dir.s
                if dir.v.z < 1e-4:
                    continue
                x_exit = x0 + dir.v.x * film_px / dir.v.z
                y_exit = y0 + dir.v.y * film_px / dir.v.z
                ix, iy = ti.cast(tm.floor(x_exit), ti.i32), ti.cast(
                    tm.floor(y_exit), ti.i32
                )
                if not (0 <= ix < W_sim and 0 <= iy < H_sim):
                    continue
                col = front_rgb_f[iy, ix]
                for l in ti.static(range(n_layers)):
                    z_l = depth_f[l]
                    x_l = x0 + dir.v.x * z_l / dir.v.z
                    y_l = y0 + dir.v.y * z_l / dir.v.z
                    ix_l = ti.cast(tm.floor(x_l), ti.i32)
                    iy_l = ti.cast(tm.floor(y_l), ti.i32)
                    dv = (
                        dens_f[l, iy_l, ix_l]
                        if (0 <= ix_l < W_sim and 0 <= iy_l < H_sim)
                        else 1.0
                    )
                    ab = absorb_f[l]
                    col *= 1.0 - ab + ab * dv
                acc += col
            bounced_rgb_f[py, px] = acc / ti.max(1, S)

    bounce(424242)
    final_rgb = np.clip(front_rgb + bounced_rgb_f.to_numpy() * back_refl, 0.0, 1.0)
else:
    final_rgb = front_rgb

# ───────────────────────── save PNG ──────────────────────────────────────────
out = linear_to_srgb(final_rgb, args.gamma)
Image.fromarray((out * 255).astype(np.uint8)).resize(
    (round(W_sim / args.supersample), args.height), Image.LANCZOS
).save(args.output)
print("\n[OK] saved →", args.output)
