import taichi as ti
from PIL import Image
import numpy as np, argparse, sys
import os

# ------------------------------------------------------------------
#  DEBUG / OOM-MESSAGE HELPER
# ------------------------------------------------------------------
import faulthandler, traceback, os, signal, sys, taichi as ti

# 1. let Python print something even on a hard crash
faulthandler.enable(all_threads=True)


# 2. print a Python traceback for every uncaught exception
def _excepthook(exc_type, exc, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        return sys.__excepthook__(exc_type, exc, tb)
    traceback.print_exception(exc_type, exc, tb)
    sys.exit(1)


sys.excepthook = _excepthook


# 3. force Taichi kernels to finish immediately so errors appear here
def safe(kernel, *args, **kwargs):
    kernel(*args, **kwargs)
    ti.sync()  # flush & wait -> get OOM right here


# ---------------- CLI ----------------
cli = argparse.ArgumentParser()
cli.add_argument("--input", required=True)
cli.add_argument("--bn", default="res/bn.png")  # 1024² LUT
cli.add_argument("-H", "--height", type=int, default=2048)
cli.add_argument("--scale", type=int, default=4)
cli.add_argument("--coverage", type=float, default=0.95)  # 0–1
cli.add_argument("--alpha", type=float, default=2.0)
cli.add_argument("--output", default="film_bw.png")
cli.add_argument(
    "--arch", default="auto", choices=["auto", "cpu", "cuda", "vulkan", "metal"]
)
args = cli.parse_args()


# ---------------- Taichi init ----------------
def pick(a):
    if a != "auto":
        return getattr(ti, a)
    try:
        return ti.gpu
    except ti.TaichiRuntimeError:
        return ti.cpu


os.environ["TI_DEVICE_MEMORY_GB"] = "4"  # <-- 4 GB budget
ti.init(
    arch=pick(args.arch),
    debug=True,  # enable extra checks
    log_level=ti.DEBUG,  # verbose logging
)

print("[Taichi] arch =", ti.lang.impl.current_cfg().arch)


# ---------------- Read / resize source ---------------
im = Image.open(args.input).convert("L")
H_scan = args.height
W_scan = round(im.width * H_scan / im.height)
im = im.resize((W_scan, H_scan), Image.LANCZOS)
scan_np = np.asarray(im, np.float32) / 255.0

# ---------------- Load blue-noise LUT ----------------
lut = Image.open(args.bn).convert("L")
if lut.size != (1024, 1024):
    print("blue-noise texture must be 1024×1024", file=sys.stderr)
    sys.exit(1)
bn_np = np.asarray(lut, np.uint8).astype(np.float32) / 255.0
BN = 1024

# ---------------- Canvas sizes -----------------------
S = args.scale
H, W = H_scan * S, W_scan * S
print(f"[Info] scan {H_scan}×{W_scan}   canvas {H}×{W}   scale {S}")

# ---------------- Fields -----------------------------
expo_lo = ti.field(ti.f32, shape=(H_scan, W_scan))
expo_lo.from_numpy(scan_np)
expo_hi = ti.field(ti.f32, shape=(H, W))
mask = ti.field(ti.u8, shape=(H, W))  # 0/1 uint8
T_hi = ti.field(ti.f32, shape=(H, W))
bn_tex = ti.field(ti.f32, shape=(BN, BN))
bn_tex.from_numpy(bn_np)


@ti.kernel
def upsample():
    for i, j in expo_hi:
        expo_hi[i, j] = expo_lo[i // S, j // S]


thr = max(0.0, min(args.coverage, 0.9999))
print(f"[Info] effective coverage threshold = {thr:.4f}")

one_u8 = ti.cast(1, ti.u8)  # pre-built literals
zero_u8 = ti.cast(0, ti.u8)


@ti.kernel
def mark():
    for i, j in mask:
        mask[i, j] = one_u8 if bn_tex[i % BN, j % BN] < thr else zero_u8


R = (S + 1) // 2


@ti.kernel
def dilate():
    for i, j in mask:
        if mask[i, j]:
            for di, dj in ti.static(ti.ndrange((-R, R + 1), (-R, R + 1))):
                if di * di + dj * dj <= R * R:
                    ii, jj = i + di, j + dj
                    if 0 <= ii < H and 0 <= jj < W:
                        mask[ii, jj] = one_u8


@ti.kernel
def shade(alpha: ti.f32):
    for i, j in T_hi:
        if mask[i, j]:
            E = expo_hi[i, j]
            D = 1 - ti.exp(-4.0 * E)
            T_hi[i, j] = ti.exp(-alpha * D)
        else:
            T_hi[i, j] = 1.0


film = ti.field(ti.f32, shape=(H_scan, W_scan))


@ti.kernel
def down():
    for i, j in film:
        acc = 0.0
        for di, dj in ti.ndrange(S, S):
            acc += T_hi[i * S + di, j * S + dj]
        film[i, j] = acc / (S * S)


# ---------------- Pipeline ---------------------------
print("[Info] processing", args.input)
upsample()
mark()
# dilate()
shade(args.alpha)
down()

out = (film.to_numpy() * 255).astype(np.uint8)
Image.fromarray(out, "L").save(args.output)
print("[OK] saved →", args.output)
