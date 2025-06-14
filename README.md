# Retraced — Physically-Based Film Emulator

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Retraced** (working title “CyberFilm”) is an open-source, Taichi-accelerated simulator that recreates the look & feel of silver-halide emulsions — and soon color films — purely in software.

| Input                      | Retraced                    |
| -------------------------- | --------------------------- |
| ![alt text](img/input.jpg) | ![alt text](img/output.png) |

---

## ✅ Highlights

1. **Physically Based**  
   • Grain nucleation & growth as a Poisson process  
   • Log-normal size distributions
2. **Low Carbon Footprint**  
   • No chemicals, no darkroom  
   • All processing on CPU/GPU
3. **Ultra-Customizable “CyberFilm”**  
   • Tweak grain radius, σ, exposure (λ_scale), sampling count  
   • Build your own film stocks, preview black-&-white now, color soon

---

## 🚀 Features

- Black-and-white silver grain emulator
- Jittered sampling antialias filter
- Taichi GPU/CPU backends (CUDA/Vulkan/Metal/CPU)
- Interactive tone-curve scanner (PyQt6)
- CLI batch-processing & real-time GUI
- Future: Multi-layer color dye coupler emulation

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

```text
taichi>=1.5.0
numpy>=1.21
pillow>=10.0
PyQt6
```

---

## ⚙️ Quickstart

1. Clone:

   ```bash
   git clone https://github.com/dannyHallo/retraced.git
   cd Retraced
   ```

2. Create & activate env:

   ```bash
   conda create -n retraced-demo python=3.12 -y
   conda activate retraced-demo
   pip install -r requirements.txt
   ```

3. Run black-&-white grain emu:

   ```bash
   python src/retraced.py \
     --input   path/to/image.jpg \
     --height  3000 \
     --samples 100 \
     --grain_radius 0.2 \
     --grain_sigma  0.05 \
     --sigma_filter 0.4 \
     --lambda_scale 3.0 \
     --output  out.png
   ```

   Produces:

   - `out_neg.png` — silver-density negative
   - `out.png` — positive print

4. Launch interactive tone-curve “scanner”:

   ```bash
   python src/scanner.py
   ```

---

## 🔮 Roadmap

- [ ] Color-film emulation (multi-layer dye couplers)
- [ ] Batch CLI & presets for popular film stocks
- [ ] Real-time viewport preview in Taichi GUI

---

## ⚖️ License

MIT © 2025 Ruitian Yang 杨瑞天

---

## 📂 References

[Image Resources](https://www.pexels.com/search/4k/)

[ADVANCED EMULSION: Silver Halide Crystals, Imaging Couplers, Orange Masks and Processing](https://www.youtube.com/watch?v=I4_7tW-cx1I)

[Film grain rendering](https://www.youtube.com/watch?v=Gj4p5cocebc)

[Shadertoy - Realistic Film Grain Rendering](https://www.shadertoy.com/view/lcXyR4)
