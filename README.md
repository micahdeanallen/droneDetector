## Jetson Orin Nano setup (JetPack 6.2 / L4T 36.5)

### ONNX Runtime

`ort`'s bundled binaries don't work here, so the library is fetched separately
and loaded at runtime via `load-dynamic`.

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
unzip -o onnxruntime_gpu-*.whl -d ort_extract
mkdir -p lib
cp ort_extract/onnxruntime/capi/libonnxruntime.so.1.20.0 lib/libonnxruntime.so
```

Add to `~/.bashrc`:
```bash
export ORT_DYLIB_PATH=~/droneDetector/lib/libonnxruntime.so
```

### Build and run

```bash
cargo build --release --features load-dynamic
./target/release/drone_detection_pipeline --source camera --jetson-cam
```

### Gotchas that cost real time

- **The `api-20` feature is mandatory.** `ort` 2.0-rc.12 supports ONNX Runtime
  1.17–1.24 via `api-*` feature flags. Without one matching your library,
  `Session::builder()` hangs silently with no error.
- **Do not call `with_optimization_level()`.** At api-20 against ORT 1.20 it
  fails with "graph_optimization_level is not valid". ORT's defaults are fine.
- **PyPI aarch64 wheels don't work** — they're built for generic ARM servers.
  Use a Jetson-native build.
- **ORT 1.24 wheels need glibc 2.38**; JetPack 6.2 ships 2.35. Stay on 1.20–1.23.

### Performance baseline


|
 Platform 
|
 Preprocess 
|
 Inference 
|
 Throughput 
|
|
---
|
---
|
---
|
---
|
|
 PC (x86, CPU) 
|
 ~16ms 
|
 ~150ms 
|
 ~6 fps 
|
|
 Jetson Orin Nano (CPU) 
|
 ~40ms 
|
 ~1000ms 
|
 ~1 fps 
|

CPU inference on the Jetson is too slow for real-time detection. CUDA execution
provider is the next step.
