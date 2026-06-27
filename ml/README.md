# AirGuard-AI — MINA Multi-Task 1D-CNN (Improved)

Pipeline Python/PyTorch phân loại mức cảnh báo chất lượng không khí từ chuỗi cảm biến MOX (UCI Air Quality), theo kiến trúc **MINA** với 4 cải tiến: transient channel, Focal Loss, window sweep, multi-task learning.

## Yêu cầu

- Python 3.10+
- PyTorch, pandas, scikit-learn, matplotlib

```bash
cd ml
pip install -r requirements.txt
```

## Chạy pipeline

```bash
# 1. Tải dataset UCI
python src/download_data.py

# 2. Sweep cửa sổ 6h vs 3h (chọn theo val macro-F1)
python src/sweep_window.py --epochs 15

# 3. Huấn luyện full (dùng window thắng từ sweep)
python src/train.py --epochs 30 --window-size 3

# 4. Đánh giá (classification + safety policies)
python src/evaluate.py

# 4b. Safety retrain (Focal Loss cực đoan) + đánh giá
python src/train.py --epochs 30 --window-size 3 --focal-preset safety --out results/best_model_safety.pt
python src/evaluate.py --ckpt results/best_model_safety.pt

# 5. Lượng tử hóa Q1.15
python src/quantize.py
```

## Cải tiến so với baseline

| Cải tiến | Mô tả |
|----------|--------|
| **Transient channel** | Kênh 9 = `mean(|Δ|)` qua 8 cảm biến — bắt gia tốc thay đổi khí |
| **Focal Loss** | Xử lý mất cân bằng lớp (gamma=2, alpha theo tần suất train) |
| **Window sweep** | So sánh W=6h vs W=3h, chọn theo val macro-F1 |
| **Multi-task** | Cls (3 lớp) + Reg (CO(GT)) với `L = L_cls + 0.3 * L_reg` |
| **SE Attention** | Squeeze-Excitation trên backbone trước dual-head |

## Kiến trúc

```
Input (9ch × W)  [8 sensor + 1 transient]
  → MINA Backbone (stem k=7 + 2 Inception 5 nhánh + residual)
  → SE Channel Attention
  → Cls Head (3 lớp)  +  Reg Head (CO ppm)
```

## Kết quả (chạy thực tế)

**Window sweep:** W=3h thắng (val macro-F1 **74.3%** vs W=6h **73.1%**)

| Metric | Baseline cũ | Improved MINA |
|--------|-------------|---------------|
| Window | 24h | **3h** |
| Input channels | 8 | **9** (+ transient) |
| Parameters | 3,071 | **4,018** |
| MACs / inference | ~34,680 | **~6,832** |
| Test accuracy | 66.3% | **70.7%** (+4.4%) |
| Test macro-F1 | 67.6% | **71.3%** (+3.7%) |
| Warning F1 | 0.57 | **0.61** |
| Danger recall | 87% | **88%** |
| CO MAE (reg) | — | **0.48 ppm** |
| Q1.15 cls drop | 0% | **0%** |

**Nhận xét:**
- Cải thiện rõ accuracy (+4.4%) và macro-F1 (+3.7%) so với baseline MINA đơn giản.
- Cửa sổ 3h phù hợp hơn 24h — transient không bị pha loãng.
- Focal Loss giúp Warning class (F1 0.61 vs 0.57 trước).
- Multi-task CO regression học được (MAE 0.48 ppm).
- Q1.15 không suy giảm độ chính xác phân loại.

Artifacts: `results/best_model.pt`, `results/window_sweep.json`, `results/eval_metrics.json`, `results/q15_weights/`.

## Safety-Critical (3 lớp phòng vệ)

Ba cải tiến **không đổi kiến trúc NPU / Q1.15** — chỉ post-processing firmware hoặc retrain loss:

| Vũ khí | Module | Cần retrain? | Mô tả |
|--------|--------|--------------|-------|
| **Asymmetric threshold** | `src/inference_policy.py` | Không | `DANGER` nếu `P(Danger) >= T` (mặc định T=0.15, sweep trên val) |
| **Focal Loss safety** | `src/losses.py`, `train.py` | Có | `--focal-preset safety` → alpha=[0.05,0.15,0.80], gamma=3 |
| **OR fusion** | `src/safety_fusion.py` | Không | `final = npu_danger OR hard_logic(PT08.S1, transient)` |

### Chạy đánh giá safety

```bash
# Phase A — checkpoint hiện tại (không retrain)
python src/evaluate.py --ckpt results/best_model.pt

# Ràng buộc precision tối thiểu khi sweep ngưỡng (val)
python src/evaluate.py --min-precision 0.55

# Phase B — model retrain với Focal safety
python src/train.py --epochs 30 --window-size 3 --focal-preset safety --out results/best_model_safety.pt
python src/evaluate.py --ckpt results/best_model_safety.pt
```

Outputs: `results/safety_metrics.json`, `results/threshold_sweep.json`, `results/danger_pr_sweep.png`, `results/confusion_matrix_fused.png`.

### Kết quả safety (test set, UCI)

**Checkpoint `best_model.pt` (Focal auto, macro-F1 tối ưu):**

| Policy | Danger Recall | Danger Precision | Safe→Danger FPR | Macro-F1 |
|--------|---------------|------------------|-----------------|----------|
| argmax (baseline) | **87.6%** | 59.8% | 0.4% | **71.3%** |
| asymmetric (T=0.15 val) | **99.7%** | 29.2% | 70.6% | 29.5% |
| hard-only | 98.6% | 27.7% | 79.0% | 25.5% |
| **fused OR** | **100%** | 26.4% | 91.5% | 19.1% |

**Checkpoint `best_model_safety.pt` (`--focal-preset safety`):**

| Policy | Danger Recall | Danger Precision | Safe→Danger FPR | Macro-F1 |
|--------|---------------|------------------|-----------------|----------|
| argmax | **97.2%** | 33.8% | 46.6% | 44.4% |
| asymmetric (T=0.33 val) | **100%** | 27.0% | 85.5% | 22.6% |
| **fused OR** | **100%** | 26.0% | 94.8% | 17.1% |

**Đạt mục tiêu recall:** asymmetric ≥93% và fused ≥95% trên cả hai checkpoint. **Trade-off:** macro-F1 và Safe→Danger FPR tăng mạnh — hệ thống an toàn ưu tiên không bỏ sót Danger.

Ngưỡng hard-logic (train Danger windows, p10): `PT08.S1 >= 1043`, `transient >= 15.58` (đơn vị gốc sau denorm).

### Firmware — asymmetric policy (C pseudocode)

Sau NPU inference, firmware đọc 3 softmax Q0.15 và áp policy fail-safe:

```c
// Class order: 0=Safe, 1=Warning, 2=Danger
// Q0.15: 0.15 ≈ 4915 (0x1333)
#define DANGER_THRESH_Q15  4915

uint8_t asymmetric_class(int16_t p_safe, int16_t p_warn, int16_t p_danger) {
    if (p_danger >= DANGER_THRESH_Q15) return CLASS_DANGER;
    return CLASS_SAFE;  // optional: elif p_warn >= WARN_THRESH
}

uint8_t fuse_alarm(uint8_t npu_class, uint8_t hard_flag) {
    uint8_t npu_danger = (npu_class == CLASS_DANGER);
    return (npu_danger | hard_flag) ? CLASS_DANGER : npu_class;
}
```

### RTL / MMIO — hard-logic path

Hard path chạy song song NPU, không dùng CO(GT):

| Register | Offset | Mô tả |
|----------|--------|-------|
| `SENSOR_S1_RAW` | `0x00` | PT08.S1 ADC (16-bit unsigned) |
| `SENSOR_TRANSIENT` | `0x04` | Gia tốc thay đổi (firmware tính `mean(|Δ|)`) |
| `HARD_THRESH_S1` | `0x08` | Ngưỡng S1 (calibrate tại chỗ) |
| `HARD_THRESH_TR` | `0x0C` | Ngưỡng transient |
| `HARD_ALARM` | `0x10` | bit0 = `(S1 >= T_s1) OR (tr >= T_tr)` |
| `NPU_RESULT` | `0x20` | Class từ MINA cls head (post-softmax policy) |
| `FINAL_ALARM` | `0x24` | bit0 = `HARD_ALARM \| NPU_DANGER` → GPIO/UART |

RTL: 2 comparator + 1 OR gate (~vài chục LUT). RISC-V đọc `FINAL_ALARM` trong ISR fail-safe.

### Rủi ro / hạn chế

1. UCI dataset ≠ công nghiệp thực — ngưỡng cần hiệu chuẩn tại chỗ.
2. Hard-logic trên MOX proxy (PT08.S1 ≠ ppm CO thật).
3. Sweep `min_precision=0` tối đa recall; dùng `--min-precision` để cân bằng FP.
4. Deploy path **không dùng CO(GT)** — chỉ dùng khi train/eval.

## CLI huấn luyện

```bash
python src/train.py \
  --window-size 3 \
  --epochs 30 \
  --loss focal \        # focal | weighted_ce | ce
  --focal-preset safety \  # alpha=[0.05,0.15,0.80], gamma=3
  --scheduler cosine \    # cosine | plateau | none
  --patience 8 \        # early stopping
  --weight-decay 1e-4 \
  --alpha 0.3 \         # trọng số regression loss
  --no-transient \      # tắt kênh transient
  --no-multi-task       # chỉ classification
```

## Khoa học & bảo vệ AI (sơ tuyển)

```bash
python src/baselines.py              # RF, LR, MLP vs MINA
python src/ablation.py --epochs 12   # ablation study
python src/run_seeds.py --seeds 42 1 7  # mean +/- std
```

Tài liệu: [`docs/BAO_VE_AI_PIPELINE.md`](docs/BAO_VE_AI_PIPELINE.md)

## Hướng phần cứng (MINA accelerator)

- PEA linh hoạt + SBA + 4 LDM/PE (xem `docs/MINA.md`)
- Q1.15 weights: `results/q15_weights/`
- SoC alert path dùng **cls head**; reg head cho giám sát nồng độ
