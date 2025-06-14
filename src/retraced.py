import argparse
import math
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image
import taichi as ti
import taichi.math as tm
import tomllib as toml

# ──────────────────────────────── CLI ─────────────────────────────────────────
cli = argparse.ArgumentParser(description="Physically-based film-grain emulator")

cli.add_argument("--input", required=True, help="RGB input image (anything PIL reads)")
cli.add_argument("--height", type=int, default=1080, help="Final image height (px)")
cli.add_argument(
    "--supersample",
    type=int,
    default=1,
    help="Oversampling factor (render at height×SS then downscale)",
)
cli.add_argument(
    "--samples", type=int, default=100, help="# Monte-Carlo samples per output pixel"
)
cli.add_argument(
    "--gamma", type=float, default=2.2, help="Electro-optical transfer γ (sRGB ≈ 2.2)"
)
cli.add_argument(
    "--film_cfg",
    default="film-config.toml",
    help="TOML file that describes the film stock",
)
cli.add_argument("--output", default="out.png", help="Destination file")
args = cli.parse_args()

# ─────────────────────────── Read film-config ────────────────────────────────
cfg: Dict = toml.loads(Path(args.film_cfg).read_text())

layers_cfg: List[Dict] = cfg.get("layer", [])
if not layers_cfg:
    raise ValueError("film-config.toml must contain at least one [[layer]]")

valid_col = {"R", "G", "B", "L"}  # L = monochrome luminance layer
valid_filt = {"red", "green", "blue", "cyan", "magenta", "yellow", None}

for i, lay in enumerate(layers_cfg):
    # --- colour ----------------------------------------------------------------
    colour = str(lay.get("color", "")).upper()
    if colour not in valid_col:
        raise ValueError(f"layer {i}: unknown colour '{colour}'")
    lay["color"] = colour

    # --- numerical defaults ----------------------------------------------------
    lay.setdefault("grain_radius", 1.25)
    lay.setdefault("grain_sigma", 0.40)
    lay.setdefault("sigma_filter", 0.80)

    # --- optional filter -------------------------------------------------------
    filt_raw = lay.get("filter")  # may be missing / None / "" / string
    filt = None if filt_raw in (None, "", "None", "none") else str(filt_raw).lower()
    if filt not in valid_filt:
        raise ValueError(f"layer {i}: unknown filter '{filt_raw}'")
    lay["filter"] = filt  # will be None when no filter is present

# ──────────────────────── Derived global sizes ───────────────────────────────
SS = max(1, args.supersample)  # supersampling factor
H_final = args.height
H_sim = H_final * SS  # off-screen buffer height

# ───────────────────────────── Init Taichi ────────────────────────────────────
ti.init(arch=ti.gpu, default_ip=ti.i32, random_seed=0)
print("[Taichi] backend:", ti.cfg.arch)


# ─────────────────────── Load & linearise input image ─────────────────────────
def srgb_to_linear(arr: np.ndarray, g: float) -> np.ndarray:
    arr = arr * (1 / 255)
    return np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** g).astype(
        np.float32
    )


pil_in = Image.open(args.input).convert("RGB")  # RGB guaranteed
W_sim = round(pil_in.width * H_sim / pil_in.height)
img_lin = srgb_to_linear(
    np.asarray(pil_in.resize((W_sim, H_sim), Image.LANCZOS), np.float32),
    args.gamma,
)  # (H, W, 3) RGB-linear
H_sim, W_sim, _ = img_lin.shape
print(f"[Info] simulation size: {W_sim}×{H_sim}  (SS×{SS})")

# ────────────────────── Helper: create per-layer source ───────────────────────
rgb_index = {"R": 0, "G": 1, "B": 2}
block_by_filter = {
    "red": (0,),
    "cyan": (0,),
    "green": (1,),
    "magenta": (1,),
    "blue": (2,),
    "yellow": (2,),
}


def luminance(rgb: np.ndarray) -> np.ndarray:
    # Rec. 709 luma coeffs
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(
        np.float32
    )


src_layers: List[np.ndarray] = []
working_rgb = img_lin.copy()  # mutable buffer that filters punch holes in
for i, lay in enumerate(layers_cfg):
    colour = lay["color"]
    if colour == "L":
        src = luminance(working_rgb)
    else:
        src = working_rgb[..., rgb_index[colour]].astype(np.float32)
    src_layers.append(src)

    # apply filter (blocks one primary) for layers BELOW this one
    filt = lay["filter"]
    if filt:
        for ch in block_by_filter[filt]:
            working_rgb[..., ch] = 0.0

# ───────────────────────── Taichi fields & buffers ────────────────────────────
src_tex = ti.field(dtype=ti.f32, shape=(H_sim, W_sim))  # current layer source
neg = ti.field(dtype=ti.f32, shape=(H_sim, W_sim))  # rendered negative layer

# per-layer physical parameters – scalar 0-D Taichi fields (cheap to update)
R_f = ti.field(dtype=ti.f32, shape=())
SIG_f = ti.field(dtype=ti.f32, shape=())
SIGF_f = ti.field(dtype=ti.f32, shape=())
R2_f = ti.field(dtype=ti.f32, shape=())
sigma_ln_f = ti.field(dtype=ti.f32, shape=())
mu_ln_f = ti.field(dtype=ti.f32, shape=())
maxR_f = ti.field(dtype=ti.f32, shape=())
lambda_fac_f = ti.field(dtype=ti.f32, shape=())
ag_f = ti.field(dtype=ti.f32, shape=())

# consts
π = math.pi
uMax = 1.0
eps = 1e-5
U32 = ti.u32
S = args.samples


# ─────────────────────── RNG helpers (unchanged) ─────────────────────────────
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
    x = U32(0)
    prod = expLam
    summ = expLam
    lim = ti.cast(ti.floor(1e4 * lam), U32)
    while (u > summ) and (x < lim):
        x += 1
        prod *= lam / ti.cast(x, ti.f32)
        summ += prod
    return Rnd(r.s, x)


@ti.func
def sq(a, b, c, d):
    return (a - c) * (a - c) + (b - d) * (b - d)


# ─────────────────────────── Render kernel ───────────────────────────────────
@ti.kernel
def render(seed: U32):
    fseed = wang(seed)
    # R = R_f[None]
    SIG = SIG_f[None]
    SIG_F = SIGF_f[None]
    R2 = R2_f[None]
    sigma_ln = sigma_ln_f[None]
    mu_ln = mu_ln_f[None]
    maxR = maxR_f[None]
    λ_fac = lambda_fac_f[None]
    ag = ag_f[None]

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
            ix = ti.cast(tm.clamp(ti.floor(xG), 0, W_sim - 1), ti.i32)
            iy = ti.cast(tm.clamp(ti.floor(yG), 0, H_sim - 1), ti.i32)

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


# ────────────────────────── Per-layer render loop ────────────────────────────
def prep_physics(r_px: float, sig_px: float, sig_filter_px: float):
    R_f[None] = r_px
    SIG_f[None] = sig_px
    SIGF_f[None] = sig_filter_px
    R2_f[None] = r_px * r_px
    if sig_px == 0:
        sigma2_ln = 0.0
        sigma_ln = 0.0
    else:
        sigma2_ln = math.log((sig_px / r_px) ** 2 + 1)
        sigma_ln = math.sqrt(sigma2_ln)
    sigma_ln_f[None] = sigma_ln
    mu_ln_f[None] = math.log(r_px) - 0.5 * sigma2_ln
    maxR = r_px if sig_px == 0 else math.exp(mu_ln_f[None] + 3.0902 * sigma_ln)
    maxR_f[None] = maxR
    ag = 1.0 / math.ceil(1.0 / r_px)
    ag_f[None] = ag
    lambda_fac_f[None] = ag * ag / (π * (r_px * r_px + sig_px * sig_px))


results: List[np.ndarray] = []
for idx, lay in enumerate(layers_cfg):
    # convert grain params from final-px → simulation-px
    r_px = lay["grain_radius"] * SS
    sig_px = lay["grain_sigma"] * SS
    sig_flt = lay["sigma_filter"] * SS
    prep_physics(r_px, sig_px, sig_flt)

    # upload source channel
    src_tex.from_numpy(src_layers[idx])

    print(
        f"[Layer {idx}] colour={lay['color']}  samples={S} "
        f"(R={lay['grain_radius']}px σ={lay['grain_sigma']})"
    )
    render(12345)
    results.append(neg.to_numpy())


# ──────────────────────────── Down-scaler ────────────────────────────────────
def linear_to_srgb(img: np.ndarray, g: float) -> np.ndarray:
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * np.power(img, 1 / g) - 0.055)


def downscale(np_img: np.ndarray) -> Image.Image:
    h = H_final
    w = round(np_img.shape[1] / SS)
    srgb = np.clip(linear_to_srgb(np_img, args.gamma), 0, 1)
    return Image.fromarray((srgb * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)


# ───────────────────────── Save composite result ─────────────────────────────
out_path = Path(args.output)
root, ext = out_path.stem, out_path.suffix or ".png"

if len(results) == 1:
    pos = 1.0 - results[0]
    downscale(pos).save(f"{root}{ext}")
else:
    neg_stack = np.stack(results, axis=-1)  # as-simulated order
    # Re-order to proper RGB: map layer colour → position
    rgb = np.zeros_like(neg_stack[..., :3])
    for lay, img in zip(layers_cfg, results):
        col = lay["color"]
        if col == "L":
            rgb += img[..., None]  # monochrome contributes equally
        else:
            rgb[..., rgb_index[col]] = img
    pos_rgb = 1.0 - rgb
    downscale(pos_rgb).save(f"{root}{ext}")

print("[OK] saved →", f"{root}{ext}")
