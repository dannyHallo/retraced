import argparse, math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import taichi as ti, taichi.math as tm
import tomllib as toml

# ──────────────────────────── CLI ────────────────────────────────────────────
cli = argparse.ArgumentParser(description="Colour-negative film-grain emulator")
cli.add_argument("--input", required=True, help="RGB input image")
cli.add_argument("--height", type=int, default=1080, help="Output height (px)")
cli.add_argument("--supersample", type=int, default=1, help="Over-sampling factor")
cli.add_argument("--samples", type=int, default=100, help="# Monte-Carlo samples/px")
cli.add_argument("--gamma", type=float, default=2.2, help="sRGB γ")
cli.add_argument(
    "--film_cfg", default="film-config.toml", help="TOML stock description"
)
cli.add_argument("--output", default="out.png", help="Destination file")
args = cli.parse_args()

# ────────────────────────── read TOML (preserve order) ───────────────────────
cfg_txt = Path(args.film_cfg).read_text(encoding="utf8")

order: List[Tuple[str, int]] = []
seen = {"emulsion": 0, "filter": 0}
for ln in cfg_txt.splitlines():
    ln = ln.strip()
    if ln.startswith("[[emulsion]]"):
        order.append(("emulsion", seen["emulsion"]))
        seen["emulsion"] += 1
    elif ln.startswith("[[filter]]"):
        order.append(("filter", seen["filter"]))
        seen["filter"] += 1

raw_cfg: Dict = toml.loads(cfg_txt)
emulsions: List[Dict] = raw_cfg.get("emulsion", [])
filters: List[Dict] = raw_cfg.get("filter", [])

if not emulsions:
    raise ValueError("At least one [[emulsion]] block is required")
if len(filters) not in {len(emulsions), len(emulsions) - 1}:
    raise ValueError("#[[filter]] must equal #[[emulsion]] or be one fewer")

EPSILON = 1e-5


def srgb_to_linear(arr, g):
    arr = arr / 255.0
    return np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** g).astype(
        np.float32
    )


def linear_to_srgb(arr, g):
    return np.where(arr <= 0.0031308, arr * 12.92, 1.055 * arr ** (1.0 / g) - 0.055)


ti.init(arch=ti.gpu, default_ip=ti.i32, random_seed=0)

# ─────────────────────────── load input ───────────────────────────────────────
pil_in = Image.open(args.input).convert("RGB")
H_sim = args.height * args.supersample
W_sim = round(pil_in.width * H_sim / pil_in.height)
img_lin = srgb_to_linear(
    np.asarray(pil_in.resize((W_sim, H_sim), Image.LANCZOS), dtype=np.float32),
    args.gamma,
)

# ─────────────────── Taichi fields & uniforms ────────────────────────────────
src_tex = ti.field(ti.f32, shape=(H_sim, W_sim))
neg = ti.field(ti.f32, shape=(H_sim, W_sim))

R_f, SIG_f, SIGF_f, R2_f = (ti.field(ti.f32, shape=()) for _ in range(4))
sigma_ln_f, mu_ln_f, maxR_f = (ti.field(ti.f32, shape=()) for _ in range(3))
lambda_fac_f, ag_f = (ti.field(ti.f32, shape=()) for _ in range(2))


# ─────────────────── random helpers and kernel ───────────────────────────────
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
def rnd01(state):  # [0,1)
    ns = xor_shift(state)
    return Rnd(ns, ns * (1.0 / 4294967295.0))


@ti.func
def rnd_gauss(state):
    r1 = rnd01(state)
    r2 = rnd01(r1.s)
    r = ti.sqrt(-2.0 * ti.log(r1.v + 1e-12))
    return Rnd(r2.s, r * ti.cos(2.0 * math.pi * r2.v))


@ti.func
def rnd_poisson(state, lam, expLam):
    r = rnd01(state)
    u = r.v
    x = ti.u32(0)
    prod = expLam
    summ = expLam
    lim = ti.cast(ti.floor(1e4 * lam), ti.u32)
    while (u > summ) and (x < lim):
        x += 1
        prod *= lam / ti.cast(x, ti.f32)
        summ += prod
    return Rnd(r.s, ti.cast(x, ti.f32))


@ti.func
def sq(a, b, c, d):
    return (a - c) * (a - c) + (b - d) * (b - d)


@ti.kernel
def render(seed: ti.u32):
    fseed = wang(seed)
    SIG = SIG_f[None]
    SIG_F = SIGF_f[None]
    R2 = R2_f[None]
    sigma_ln = sigma_ln_f[None]
    mu_ln = mu_ln_f[None]
    maxR = maxR_f[None]
    λ_fac = lambda_fac_f[None]
    ag = ag_f[None]

    for py, px in neg:
        st = wang(ti.u32(py * 73856093 ^ px * 19349663 ^ fseed))
        hit = 0.0
        for _ in range(args.samples):
            g1 = rnd_gauss(st)
            st = g1.s
            g2 = rnd_gauss(st)
            st = g2.s
            xG = ti.cast(px, ti.f32) + SIG_F * g1.v
            yG = ti.cast(py, ti.f32) + SIG_F * g2.v
            ix = ti.cast(tm.clamp(ti.floor(xG), 0, W_sim - 1), ti.i32)
            iy = ti.cast(tm.clamp(ti.floor(yG), 0, H_sim - 1), ti.i32)

            u = tm.clamp(src_tex[iy, ix], 0.0, 1.0 - EPSILON)
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
        neg[py, px] = 1.0 - hit / args.samples  # 1 → clear, 0 → opaque


# ───────────────── grain-physics helper ───────────────────────────────────────
def prep_physics(r_px, sig_px, sig_filter_px):
    R_f[None] = r_px
    SIG_f[None] = sig_px
    SIGF_f[None] = sig_filter_px
    R2_f[None] = r_px * r_px

    if sig_px == 0:
        sigma_ln = sigma2_ln = 0.0
    else:
        sigma2_ln = math.log((sig_px / r_px) ** 2 + 1)
        sigma_ln = math.sqrt(sigma2_ln)
    sigma_ln_f[None], mu_ln_f[None] = sigma_ln, math.log(r_px) - 0.5 * sigma2_ln

    maxR_f[None] = r_px if sig_px == 0 else math.exp(mu_ln_f[None] + 3.0902 * sigma_ln)
    ag = 1.0 / math.ceil(1.0 / r_px) if r_px > 0 else 1.0
    ag_f[None] = ag
    lambda_fac_f[None] = (
        ag * ag / (math.pi * (r_px * r_px + sig_px * sig_px)) if r_px > 0 else 0.0
    )


# ───────────────────────── simulation loop ───────────────────────────────────
ray = img_lin.copy()  # through-going light
results: List[Tuple[np.ndarray, np.ndarray]] = []  # (absorption-weights, density)

for idx, (kind, k) in enumerate(order):
    if kind == "emulsion":
        emu = emulsions[k]
        dye = np.array(emu["sensitising_dye_color"], dtype=np.float32) / 255.0

        # absorption weights: complement of the dye colour, normalised
        absorb = 1.0 - dye
        if absorb.sum() < EPSILON:
            raise ValueError("Sensitising dye colour must not be pure white")
        # normalization
        absorb = absorb / absorb.sum()

        # exposure: luminance seen through the absorption spectrum
        src = 1.0 - np.clip(np.tensordot(ray, absorb, axes=([-1], [0])), 0.0, 1.0)
        src_tex.from_numpy(src.astype(np.float32))

        r_px = emu.get("grain_radius", 0.02) * args.supersample
        sig_px = emu.get("grain_sigma", 0.00) * args.supersample
        sig_f = emu.get("sigma_filter", 0.08) * args.supersample
        prep_physics(r_px, sig_px, sig_f)

        print(
            f"[emu {idx}] dye={emu['sensitising_dye_color']}  "
            f"R={r_px/args.supersample:.3f}px σ={sig_px/args.supersample:.3f}px"
        )
        render(12345)
        developed_negative = neg.to_numpy()
        results.append((absorb, developed_negative))

        # attenuate spectral components for deeper layers
        #   factor = 1 - w*(1 - density)
        ray *= 1.0 - absorb[None, None, :] * (1.0 - developed_negative[..., None])
    else:  # filter
        fil = filters[k]
        fcol = np.array(fil["color"], dtype=np.float32) / 255.0
        ray *= fcol[None, None, :]
        print(f"[filter {idx}] colour={fil['color']}")

# ───────────────────────── assemble final negative ───────────────────────────
pos_rgb = np.ones_like(img_lin)
for absorb, density in results:
    layer = 1.0 - absorb[None, None, :] + absorb[None, None, :] * density[..., None]
    pos_rgb *= layer  # successive dye layers multiply

# ───────────────────────── down-scale & save ─────────────────────────────────
out = linear_to_srgb(pos_rgb, args.gamma)
out = np.clip(out, 0, 1)
out = Image.fromarray((out * 255).astype(np.uint8)).resize(
    (round(W_sim / args.supersample), args.height), Image.LANCZOS
)
out_path = Path(args.output)
out.save(out_path)
print("[OK] saved →", out_path)
