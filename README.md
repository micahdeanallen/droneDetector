# Drone Detection Pipeline

Real-time drone detection on a Jetson Orin Nano, written in Rust. Capture →
YOLOv8s inference (ONNX Runtime) → multi-object tracking → GPS-anchored output.
Classifies drone / bird / airplane / helicopter so that non-drone aerial objects
are rejected rather than counted.

**Hardware:** Jetson Orin Nano Super Dev Kit, Arducam AR0234 (CAM1), USB GPS
**Software:** JetPack 6.2 / L4T 36.5, CUDA 12.6, cuDNN 9.3, TensorRT 10.3, glibc 2.35

---

## Setup

### ONNX Runtime

`ort`'s bundled binaries don't work here, so the library is fetched separately
and loaded at runtime via `load-dynamic`. **All four shared objects are
required** — the CUDA execution provider lives in a separate library that ORT
`dlopen`s at runtime from the same directory as the core library.

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
unzip -o onnxruntime_gpu-*.whl -d ort_extract
mkdir -p lib
cp ort_extract/onnxruntime/capi/libonnxruntime.so.1.20.0        lib/libonnxruntime.so
cp ort_extract/onnxruntime/capi/libonnxruntime_providers_cuda.so     lib/
cp ort_extract/onnxruntime/capi/libonnxruntime_providers_tensorrt.so lib/
cp ort_extract/onnxruntime/capi/libonnxruntime_providers_shared.so   lib/
```

Expected sizes — a mismatch means the wrong wheel:

| File | Bytes |
|---|---|
| `libonnxruntime.so` | 21,819,208 |
| `libonnxruntime_providers_cuda.so` | 348,391,488 |
| `libonnxruntime_providers_tensorrt.so` | 848,088 |
| `libonnxruntime_providers_shared.so` | 8,168 |

This build targets `sm_87` only, requires glibc ≤ 2.34, and links CUDA 12 /
cuDNN 9 / TensorRT 10 — correct for JetPack 6.2.

Add to `~/.bashrc`:

```bash
export ORT_DYLIB_PATH=$HOME/droneDetector/lib/libonnxruntime.so
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_CACHE_MAXSIZE=2147483648
export CUDA_CACHE_PATH=$HOME/.nv/ComputeCache
```

The CUDA cache variables matter: the provider ships PTX only, no precompiled
SASS, so the driver JIT-compiles every kernel on first run (30–60s of apparent
hang). The cache makes that a one-time cost.

### Build and run

```bash
cargo build --release --features load-dynamic
sudo jetson_clocks                    # pin clocks before taking measurements
./target/release/drone_detection_pipeline --source camera --jetson-cam --cuda --profile
```

`--cuda` is a **runtime flag**, not a cargo feature. Do not build with
`--features cuda` — that maps to `ort/cuda` → `ort-sys/cuda` and reintroduces
the static link strategy that `load-dynamic` exists to avoid.

---

## Hardware-accelerated capture

Frames are scaled and colour-converted on the Tegra VIC via `nvvidconv` before
they reach the CPU, so the CPU only ever handles a letterbox-sized buffer:

```
v4l2src ! UYVY 1920x1200 ! nvvidconv ! NVMM NV12 640x400 ! nvvidconv ! RGBA ! appsink
```

The VIC target is derived from `detect::INPUT_SIZE` in `main.rs`, so changing
the input size automatically changes the hardware scale target — they cannot
drift apart. File sources keep the CPU path (`videoconvert` + `image` resize),
so backtests are unaffected by this change.

---

## Performance

Measured on the Jetson against the live camera, clocks pinned.

| Config | Preprocess | Inference | Total | Throughput |
|---|---|---|---|---|
| PC (x86, CPU), 960 | ~16ms | ~150ms | — | ~6 fps |
| Jetson CPU, 960 | ~40ms | ~1000ms | ~1040ms | ~1 fps |
| Jetson CUDA, 960 | ~50ms | ~62ms | ~112ms | ~9 fps |
| Jetson CUDA, 640 | ~47ms | ~51ms | ~98ms | ~10 fps |
| Jetson CUDA, 960 + VIC capture | 2.65ms | 46.1ms | 48.9ms | ~20 fps |
| **Jetson CUDA, 640 + VIC capture** | **1.17ms** | **23.4ms** | **24.6ms** | **~40 fps** |

**42× end-to-end improvement.** Three distinct wins, found by profiling rather
than guessing:

1. **CUDA execution provider** — 1000ms → 62ms inference. The single largest
   step, and the one that made real-time viable at all.
2. **Input size 960 → 640** — smaller effect than expected (62ms → 51ms), which
   was itself the clue that preprocessing was dominated by the 1920×1200
   *source*, not the inference *target*.
3. **VIC hardware capture** — 47ms → 1.17ms preprocessing, a 40× drop. Removed
   two full CPU passes over 2.3M pixels per frame. Freeing the CPU also cut
   inference (51ms → 23.4ms), since CUDA's host-side work was contending with a
   saturated CPU.

At 24.6ms the pipeline is **capture-bound**: it can sustain ~40 fps against a
30 fps camera, with ~8ms of slack per frame and no dropped frames.

---

## Mandatory Configurations that Cost Real Time

- **`api-20` is mandatory.** `ort` 2.0-rc.12 supports ORT 1.17–1.24 via `api-*`
  feature flags. Without one matching your library, `Session::builder()` hangs
  silently with no error.
- **Do not call `with_optimization_level()`.** At api-20 against ORT 1.20 it
  fails with "graph_optimization_level is not valid". ORT's defaults are fine.
- **`ldd libonnxruntime.so` showing no CUDA dependencies is expected**, not a
  sign of a CPU-only build. The core library is deliberately CUDA-free; the
  provider is `dlopen`ed at runtime. Checking with `ldd` and concluding "CPU
  build" sends you hunting for a library you already have.
- **PyPI aarch64 wheels don't work** — built for generic ARM servers, they hang
  at session creation. Use a Jetson-native build.
- **ORT 1.24 wheels need glibc 2.38**; JetPack 6.2 ships 2.35. Stay on 1.20–1.23.
- **Don't gate execution providers on `cfg!(feature = ...)`.** Cargo features
  and runtime EP selection are different axes; tying them together means a
  `load-dynamic` build silently never registers CUDA and falls back to CPU with
  no error. Use `.error_on_failure()` while debugging so registration failures
  are loud.
- **GStreamer caps cannot contain spaces.** `gst_parse_launch` splits on
  whitespace, so `video/x-raw(memory:NVMM), format=NV12` parses as two elements
  and fails. Also note it's `NVMM`, not `NVVM`.
- **Pin clocks before measuring.** Without `jetson_clocks`, GPU DVFS produced a
  39–67ms spread on identical work. Pinned, the same work runs 23.37–23.53ms.
- **Indoor testing produces false positives.** The model is trained on aerial
  datasets and expects sky backgrounds; wall decals and monitor edges read as
  aerial objects. Validate against footage, not the room.
- **Always shut down with `sudo shutdown -h now`.** Interrupted writes on the
  SD card cause journal corruption and read-only remounts, which present as
  phantom code regressions — reverts that don't apply, rebuilds that don't run.

---

