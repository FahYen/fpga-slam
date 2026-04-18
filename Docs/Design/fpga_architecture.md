# FPGA Architecture for RangeNet53 Inference

## 1. Constraints

### Platform: AWS F1 — Xilinx VU9P


| Resource      | Amount                             |
| ------------- | ---------------------------------- |
| DSP48E2       | 6,840                              |
| LUTs          | ~1,182K                            |
| FFs           | ~2,364K                            |
| BRAM (36Kb)   | 4,320 → **9.5 MB**                 |
| URAM (288Kb)  | 960 → **33.75 MB**                 |
| On-chip total | **~43 MB**                         |
| DDR4          | 4 channels, **~64 GB/s** aggregate |


### Network: RangeNet53

- Backbone: DarkNet53 — 5 encoder stages, 5 decoder stages, 1 head.
- Encoder residual-block counts: `[1, 2, 8, 8, 4]` (23 blocks, 46 block-convs + 5 downsamples + stem = **52 conv layers**, plus **23 residual adds**).
- Decoder: 5 stages, each with 1 upsample, 1 skip-add, and 1 residual block = **15 conv layers** and **10 element-wise adds** total in the decoder.
- Head: 1 × `Conv2d(32 → n_classes, 3×3)` (`n_classes=20` for SemanticKITTI).
- All downsampling is horizontal-only (`stride=[1,2]`). Height stays fixed at **64** throughout.
- Encoder residual block: `1×1 (C→C/2)` → BN → LeakyReLU → `3×3 (C/2→C)` → BN → LeakyReLU → skip-add.
- Decoder stage: `ConvTranspose2d(2C→C, kernel=[1,4], stride=[1,2])` → BN → LeakyReLU → encoder skip-add → `1×1 (C→2C)` → BN → LeakyReLU → `3×3 (2C→C)` → BN → LeakyReLU → residual add.

#### Layer channel and kernel schedule

Assumptions for the table below:

- input tensor is `[5, 64, 2048]`
- backbone output stride is `32`
- segmentation head uses `20` classes
- `Output H×W` below means the spatial size after that operation
- `Total MACs` is computed as `Repeat × H_out × W_out × C_out × K_h × K_w × C_in` for conv/upconv rows
- BN + LeakyReLU follow every conv / transposed-conv in the software model, but are omitted from the table to keep it readable

##### Stem + encoder


| Stage | Repeat | Operation       | Channels             | Kernel / stride | Output H×W | Total MACs (row) |
| ----- | ------ | --------------- | -------------------- | --------------- | ---------- | ---------------- |
| Stem  | 1      | conv            | `5 → 32`             | `3×3 / [1,1]`   | `64×2048`  | `188,743,680`    |
| Enc1  | 1      | downsample conv | `32 → 64`            | `3×3 / [1,2]`   | `64×1024`  | `1,207,959,552`  |
| Enc1  | 1      | block conv1     | `64 → 32`            | `1×1 / [1,1]`   | `64×1024`  | `134,217,728`    |
| Enc1  | 1      | block conv2     | `32 → 64`            | `3×3 / [1,1]`   | `64×1024`  | `1,207,959,552`  |
| Enc1  | 1      | residual add    | `64 + 64 → 64`       | `-`             | `64×1024`  | `-`              |
| Enc2  | 1      | downsample conv | `64 → 128`           | `3×3 / [1,2]`   | `64×512`   | `2,415,919,104`  |
| Enc2  | 2      | block conv1     | `128 → 64`           | `1×1 / [1,1]`   | `64×512`   | `536,870,912`    |
| Enc2  | 2      | block conv2     | `64 → 128`           | `3×3 / [1,1]`   | `64×512`   | `4,831,838,208`  |
| Enc2  | 2      | residual add    | `128 + 128 → 128`    | `-`             | `64×512`   | `-`              |
| Enc3  | 1      | downsample conv | `128 → 256`          | `3×3 / [1,2]`   | `64×256`   | `4,831,838,208`  |
| Enc3  | 8      | block conv1     | `256 → 128`          | `1×1 / [1,1]`   | `64×256`   | `4,294,967,296`  |
| Enc3  | 8      | block conv2     | `128 → 256`          | `3×3 / [1,1]`   | `64×256`   | `38,654,705,664` |
| Enc3  | 8      | residual add    | `256 + 256 → 256`    | `-`             | `64×256`   | `-`              |
| Enc4  | 1      | downsample conv | `256 → 512`          | `3×3 / [1,2]`   | `64×128`   | `9,663,676,416`  |
| Enc4  | 8      | block conv1     | `512 → 256`          | `1×1 / [1,1]`   | `64×128`   | `8,589,934,592`  |
| Enc4  | 8      | block conv2     | `256 → 512`          | `3×3 / [1,1]`   | `64×128`   | `77,309,411,328` |
| Enc4  | 8      | residual add    | `512 + 512 → 512`    | `-`             | `64×128`   | `-`              |
| Enc5  | 1      | downsample conv | `512 → 1024`         | `3×3 / [1,2]`   | `64×64`    | `19,327,352,832` |
| Enc5  | 4      | block conv1     | `1024 → 512`         | `1×1 / [1,1]`   | `64×64`    | `8,589,934,592`  |
| Enc5  | 4      | block conv2     | `512 → 1024`         | `3×3 / [1,1]`   | `64×64`    | `77,309,411,328` |
| Enc5  | 4      | residual add    | `1024 + 1024 → 1024` | `-`             | `64×64`    | `-`              |
| **Total** |  |  |  |  |  | **`259,094,740,992` (~259.1 GMAC)** |


##### Decoder + head


| Stage | Repeat | Operation                   | Channels          | Kernel / stride         | Output H×W | Total MACs (row) |
| ----- | ------ | --------------------------- | ----------------- | ----------------------- | ---------- | ---------------- |
| Dec5  | 1      | upconv                      | `1024 → 512`      | `1×4 transpose / [1,2]` | `64×128`   | `17,179,869,184` |
| Dec5  | 1      | skip add (from enc4 output) | `512 + 512 → 512` | `-`                     | `64×128`   | `-`              |
| Dec5  | 1      | block conv1                 | `512 → 1024`      | `1×1 / [1,1]`           | `64×128`   | `4,294,967,296`  |
| Dec5  | 1      | block conv2                 | `1024 → 512`      | `3×3 / [1,1]`           | `64×128`   | `38,654,705,664` |
| Dec5  | 1      | residual add                | `512 + 512 → 512` | `-`                     | `64×128`   | `-`              |
| Dec4  | 1      | upconv                      | `512 → 256`       | `1×4 transpose / [1,2]` | `64×256`   | `8,589,934,592`  |
| Dec4  | 1      | skip add (from enc3 output) | `256 + 256 → 256` | `-`                     | `64×256`   | `-`              |
| Dec4  | 1      | block conv1                 | `256 → 512`       | `1×1 / [1,1]`           | `64×256`   | `2,147,483,648`  |
| Dec4  | 1      | block conv2                 | `512 → 256`       | `3×3 / [1,1]`           | `64×256`   | `19,327,352,832` |
| Dec4  | 1      | residual add                | `256 + 256 → 256` | `-`                     | `64×256`   | `-`              |
| Dec3  | 1      | upconv                      | `256 → 128`       | `1×4 transpose / [1,2]` | `64×512`   | `4,294,967,296`  |
| Dec3  | 1      | skip add (from enc2 output) | `128 + 128 → 128` | `-`                     | `64×512`   | `-`              |
| Dec3  | 1      | block conv1                 | `128 → 256`       | `1×1 / [1,1]`           | `64×512`   | `1,073,741,824`  |
| Dec3  | 1      | block conv2                 | `256 → 128`       | `3×3 / [1,1]`           | `64×512`   | `9,663,676,416`  |
| Dec3  | 1      | residual add                | `128 + 128 → 128` | `-`                     | `64×512`   | `-`              |
| Dec2  | 1      | upconv                      | `128 → 64`        | `1×4 transpose / [1,2]` | `64×1024`  | `2,147,483,648`  |
| Dec2  | 1      | skip add (from enc1 output) | `64 + 64 → 64`    | `-`                     | `64×1024`  | `-`              |
| Dec2  | 1      | block conv1                 | `64 → 128`        | `1×1 / [1,1]`           | `64×1024`  | `536,870,912`    |
| Dec2  | 1      | block conv2                 | `128 → 64`        | `3×3 / [1,1]`           | `64×1024`  | `4,831,838,208`  |
| Dec2  | 1      | residual add                | `64 + 64 → 64`    | `-`                     | `64×1024`  | `-`              |
| Dec1  | 1      | upconv                      | `64 → 32`         | `1×4 transpose / [1,2]` | `64×2048`  | `1,073,741,824`  |
| Dec1  | 1      | skip add (from stem output) | `32 + 32 → 32`    | `-`                     | `64×2048`  | `-`              |
| Dec1  | 1      | block conv1                 | `32 → 64`         | `1×1 / [1,1]`           | `64×2048`  | `268,435,456`    |
| Dec1  | 1      | block conv2                 | `64 → 32`         | `3×3 / [1,1]`           | `64×2048`  | `2,415,919,104`  |
| Dec1  | 1      | residual add                | `32 + 32 → 32`    | `-`                     | `64×2048`  | `-`              |
| Head  | 1      | classifier conv             | `32 → 20`         | `3×3 / [1,1]`           | `64×2048`  | `754,974,720`    |
| **Total** |  |  |  |  |  | **`117,255,962,624` (~117.3 GMAC)** |

**Grand total MACs (conv + upconv + head): `376,350,703,616` (~376.4 GMAC)**.


### GPU reference performance (64×2048, from paper Table II)


| Platform                 | CNN (ms) | kNN post-proc (ms) | Total latency (ms) | Throughput (FPS) |
| ------------------------ | -------- | ------------------ | ------------------ | ---------------- |
| Quadro P6000 (~250W)     | 75       | 7                  | 82                 | 12               |
| Jetson AGX Xavier (~30W) | 153      | 35                 | 188                | 5                |


### Budget problem


| Item                                     | Size (INT8) |
| ---------------------------------------- | ----------- |
| Weights                                  | ~50 MB      |
| Skip connection buffers                  | ~20 MB      |
| Line buffers, partials, BN params, FIFOs | ~5+ MB      |
| **Total**                                | **~75 MB**  |
| On-chip capacity                         | ~43 MB      |


Weights alone exceed on-chip SRAM, so a fully spatial (weights-on-chip) pipeline is not possible. The architecture must stream weights from DDR.

### Activation size invariant

Every stage produces the same activation volume. Width halves while channels double, so:

```
height × width × channels = 64 × (2048 / 2^i) × (32 × 2^i) = 4 MB  (constant)
```

This is a useful property: every inter-stage buffer is exactly **4 MB** regardless of position in the network.

---

