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


ti.init(arch=pick(args.arch), random_seed=0, default_ip=ti.i32)
print("[Taichi] arch:", ti.cfg.arch)

# --------------------------------------------------------------
#  Load source image
# --------------------------------------------------------------
src_im = Image.open(args.input).convert("L")
H, W = args.height, round(src_im.width * args.height / src_im.height)
src_np = np.asarray(src_im.resize((W, H), Image.LANCZOS), np.float32) / 255.0

# --------------------------------------------------------------
#  Blue-noise textures（先保留，将来可能会用）
# --------------------------------------------------------------
BN = 1024
vec3_bn_tex = ti.field(ti.f32, shape=(BN, BN))
vec3_bn_tex.from_numpy(
    np.asarray(Image.open("res/vec3.png").convert("L"), np.float32) / 255.0
)

# --------------------------------------------------------------
#  Fields
# --------------------------------------------------------------
expo = ti.field(ti.f32, shape=(H, W))  # incoming exposure
grain = ti.field(ti.u8, shape=(H, W))  # 0/1 grain mask (现全 1)
latent = ti.field(ti.f32, shape=(H, W))  # stored energy
film = ti.field(ti.f32, shape=(H, W))  # final transmission
expo.from_numpy(src_np)

# 常量
N_RAYS = args.rays
FILM_Z = 1.0
one_u8 = ti.cast(1, ti.u8)

# --------------------------------------------------------------
#  Sobol + Owen Scramble 实现
# --------------------------------------------------------------
U = ti.u32  # shorthand


@ti.func
def reverse_bits(x: ti.u32) -> ti.u32:
    x = ((x & U(0xAAAAAAAA)) >> 1) | ((x & U(0x55555555)) << 1)
    x = ((x & U(0xCCCCCCCC)) >> 2) | ((x & U(0x33333333)) << 2)
    x = ((x & U(0xF0F0F0F0)) >> 4) | ((x & U(0x0F0F0F0F)) << 4)
    x = ((x & U(0xFF00FF00)) >> 8) | ((x & U(0x00FF00FF)) << 8)
    return (x >> 16) | (x << 16)


@ti.func
def sobol_1d(n: ti.u32) -> ti.u32:  # Van-der-Corput
    v = U(0x80000000)
    r = U(0)
    while n != 0:
        if n & 1:
            r ^= v
        n >>= 1
        v >>= 1
    return r


@ti.func
def owen_hash(x: ti.u32, seed: ti.u32) -> ti.u32:  # Nathan Vegdahl hash
    x ^= x * U(0x3D20ADEA)
    x += seed
    x *= (seed >> 16) | 1
    x ^= x * U(0x05526C56)
    x ^= x * U(0x53A22864)
    return x


@ti.func
def owen_scramble(x: ti.u32, seed: ti.u32) -> ti.u32:
    x = reverse_bits(x)
    x = owen_hash(x, seed)
    return reverse_bits(x)


###> 便捷采样函数：返回 float [0,1)
@ti.func
def sobol_owen_f32(idx: ti.u32, seed: ti.u32) -> ti.f32:
    p = sobol_1d(idx)
    p = owen_scramble(p, seed)
    return ti.cast(p, ti.f32) * (1.0 / 4294967295.0)


# --------------------------------------------------------------
#  Kernels
# --------------------------------------------------------------
@ti.kernel
def place_grains():  # 现在所有像素都放置晶粒
    for i, j in grain:
        grain[i, j] = one_u8


@ti.kernel
def raytrace():
    for py, px in expo:
        E_src = expo[py, px]
        seed = ti.u32(py * 73856093) ^ ti.u32(px * 19349663)

        for r in range(N_RAYS):
            bn_val = sobol_owen_f32(ti.u32(r), seed)
            depth = bn_val * FILM_Z
            deposited = E_src * ti.exp(-depth) / N_RAYS
            ti.atomic_add(latent[py, px], deposited)


@ti.kernel
def develop(alpha: ti.f32):
    for i, j in film:
        T = ti.exp(-alpha * latent[i, j])  # grain 始终存在
        film[i, j] = T


# --------------------------------------------------------------
#  测试：生成 1920×1080 Sobol-Owen 分布图
# --------------------------------------------------------------
TEST_W, TEST_H = 1920, 1080
test_img = ti.field(ti.f32, shape=(TEST_H, TEST_W))

# --------------------------------------------------------------
#  Pipeline
# --------------------------------------------------------------
print("[Info] simulating", f"{H}×{W}", "#rays=", N_RAYS)
place_grains()
raytrace()
develop(args.alpha)

out = (film.to_numpy() * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(out, "L").save(args.output)
print("[OK] saved →", args.output)
