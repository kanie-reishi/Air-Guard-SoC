# Bảo vệ ý tưởng AI — AirGuard-AI Pipeline

**Dự án:** AirGuard-AI SoC — Giám sát khí thải Edge AI  
**Vòng:** Sơ tuyển — Cuộc thi Thiết kế Vi mạch cho Đô thị Thông minh lần 3  
**Ngày:** 27/06/2026

---

## 1. Tóm tắt điều hành (30 giây)

Chúng tôi xây dựng **TinyML trên chip** cho cảnh báo khí thải công nghiệp:

- **Input:** 9 kênh cảm biến MOX × cửa sổ 3 giờ (8 sensor + 1 transient)
- **Model:** MINA Multi-Task 1D-CNN — **4.018 params**, **~6.832 MACs**
- **Output:** 3 lớp Safe / Warning / Danger + dự đoán CO (multi-task)
- **Deploy:** Q1.15 fixed-point, **0% accuracy drop** so với float
- **Safety:** 3 lớp phòng vệ (asymmetric threshold + hard-logic OR)

**Điểm mạnh khoa học:** có baseline đối chứng, ablation study, đa seed, training ổn định (AdamW + scheduler + early stopping).

---

## 2. Vì sao chọn bài toán này?

### 2.1. Bài toán thực tế

| Vấn đề | Giải pháp AirGuard |
|--------|-------------------|
| Cloud latency + privacy | Edge inference trên SoC |
| Sensor drift theo thời gian | Roadmap Sign-SGD + calibration |
| Cần phản hồi nhanh | < 100 ms end-to-end |
| Tiết kiệm năng lượng IoT | 4K params, Q1.15, < 5 mW target |
| Không được bỏ sót nguy hiểm | Safety-critical 3 lớp |

### 2.2. Vì sao chuỗi thời gian → 1D-CNN?

- Dữ liệu cảm biến MOX là **chuỗi đa kênh theo thời gian** (1 mẫu/giờ).
- CNN 1D học **mẫu temporal** (xu hướng, spike) mà không cần feature engineering thủ công.
- So với LSTM/Transformer: **ít params hơn 10–100×**, phù hợp deploy lên MINA accelerator.

### 2.3. Vì sao MINA (không phải generic CNN)?

MINA (IEEE TCAS-I 2024) thiết kế **hardware-aware**:

| Tiêu chí | Generic 1D-CNN | MINA |
|----------|----------------|------|
| Kiến trúc | Conv stack tuần tự | Inception 5 nhánh + residual |
| Params | ~3–11K | **4.018** |
| MACs (W=3) | ~35K | **6.832** |
| Map RTL | Khó (nhiều kernel size) | PEA + SBA native Inception |
| Q1.15 | Cần verify | **Đã verify 0% drop** |

---

## 3. Pipeline ML hiện tại

```
UCI Air Quality CSV
  → clean + interpolate missing (-200)
  → 8 MOX + 3 env channels
  → transient ch9 = mean(|Δ|) across sensors
  → sliding window W=3h, stride=1
  → label 3 lớp từ CO(GT) percentile (train only)
  → z-score normalize (train stats)
  → MINA MultiTask1D train (Focal Loss + CO reg)
  → evaluate (argmax / asymmetric / fused OR)
  → quantize Q1.15 → export weights
```

**Artifacts:** `ml/results/best_model.pt`, `q15_weights/`, `safety_metrics.json`

---

## 4. Cải tiến training (mới)

Để tăng **tính khoa học và tái lập**, `train.py` đã được nâng cấp:

| Cải tiến | Mô tả |
|----------|--------|
| **AdamW** | Weight decay 1e-4, generalization tốt hơn Adam |
| **LR Scheduler** | CosineAnnealing (default) hoặc ReduceLROnPlateau |
| **Early stopping** | Patience=8, tránh overfit epoch cuối |
| **Smooth checkpoint** | Chọn model theo moving average val macro-F1 (window=3) |
| **Danger recall** | Log thêm trong `evaluate_epoch()` |

CLI mới:
```bash
python src/train.py --weight-decay 1e-4 --scheduler cosine --patience 8 --smooth-window 3
```

---

## 5. Baseline đối chứng

**Mục đích:** Chứng minh CNN có giá trị so với ML cổ điển trên cùng features.

Chạy: `python src/baselines.py` → `results/baselines.json`

### Kết quả (test set, W=3, seed=42)

| Model | Accuracy | Macro-F1 | Danger Recall | Params / MACs |
|-------|----------|----------|---------------|---------------|
| Majority class | 25.6% | 13.6% | 100%* | — |
| Logistic Regression | **74.4%** | **75.0%** | 89.8% | ~27 features |
| Random Forest | **76.6%** | **76.9%** | 86.2% | ~200 trees |
| MLP (1 hidden) | 69.5% | 69.6% | 90.1% | ~2K params |
| **MINA CNN** | **70.7%** | **71.3%** | **87.6%** | **4.018 / 6.832** |

\*Majority luôn predict Warning (lớp đông nhất test) → Danger recall giả.

### Cách giải trình trước hội đồng

> "Random Forest đạt accuracy cao hơn ~6% trên tabular features, nhưng **không deploy được lên chip**: cần hàng trăm cây quyết định, không lượng tử hóa Q1.15, không chạy trên MINA accelerator. MINA đạt **71.3% macro-F1 với chỉ 4K params và 6.8K MACs** — trade-off chấp nhận được cho Edge AI. Thêm lớp safety (asymmetric + OR) đưa Danger Recall lên **99.7–100%** mà không cần đổi model."

---

## 6. Ablation study

**Mục đích:** Tách biệt đóng góp từng cải tiến.

Chạy: `python src/ablation.py --epochs 12` → `results/ablation.json`

### Kết quả (test set, 12 epochs, seed=42, training mới)

| Cấu hình | Thay đổi | Test Acc | Macro-F1 | Danger Recall | MACs |
|----------|----------|----------|----------|---------------|------|
| baseline_8ch_w24_ce | 8ch, W=24, CE | 61.1% | 60.7% | 92.4% | 35,520 |
| + transient | 9ch | 59.0% | 58.3% | 91.3% | 36,192 |
| + focal | Focal Loss | 57.1% | 57.0% | 93.0% | 36,192 |
| **+ W=3** | **Window 3h** | **69.6%** | **70.3%** | 90.1% | **6,832** |
| + multi-task | CO regression | 70.3% | 70.7% | 90.4% | 6,832 |
| full_improved | = multi-task | 70.3% | 70.7% | 90.4% | 6,832 |
| W=6 (control) | Window 6h | 71.8% | 72.0% | 86.5% | 9,768 |

### Nhận xét ablation

1. **Window W=3 là cải tiến lớn nhất:** macro-F1 tăng **+10 điểm** (60.7% → 70.3%) đồng thời MACs giảm **5×**.
2. Transient + Focal riêng lẻ trên W=24 không cải thiện rõ — cần kết hợp với W=3.
3. Multi-task CO regression cải thiện nhẹ (+0.4% F1) và cho phép giám sát nồng độ.
4. W=6 cho accuracy cao hơn chút (+1.1%) nhưng MACs cao hơn 43% — chọn W=3 cho ưu tiên **low area**.

---

## 7. Đa seed / error bars

**Mục đích:** Chứng minh kết quả **tái lập**, không phụ thuộc 1 seed may mắn.

Chạy: `python src/run_seeds.py --seeds 42 1 7 --epochs 20` → `results/seed_robustness.json`

### Kết quả (W=3, full config, training mới)

| Seed | Test Acc | Macro-F1 | Danger Recall | Best Epoch |
|------|----------|----------|---------------|------------|
| 42 | 71.3% | 71.8% | 88.7% | 7 |
| 1 | 67.0% | 67.7% | 91.7% | 8 |
| 7 | 69.6% | 69.8% | 87.9% | 14 |

### Tổng hợp (n=3)

| Metric | Mean ± Std |
|--------|------------|
| Test Accuracy | **69.3% ± 1.8%** |
| Macro-F1 | **69.8% ± 1.7%** |
| Danger Recall | **89.4% ± 1.7%** |
| Val Macro-F1 | **74.3% ± 1.0%** |

**Giải trình:** Std ~1.7% cho thấy pipeline **ổn định** sau khi thêm AdamW + scheduler + early stopping. Checkpoint cũ (best_model.pt, seed 42 only) đạt 70.7% — nằm trong khoảng mean+std.

---

## 8. Safety-critical (bổ sung cho bảo vệ)

Ngoài accuracy, hệ thống an toàn có **3 lớp phòng vệ** (post-processing, không đổi model):

| Chế độ | Danger Recall | Safe→Danger FPR | Macro-F1 |
|--------|---------------|-----------------|----------|
| argmax (cân bằng) | 87.6% | 0.4% | **71.3%** |
| asymmetric (T=0.15) | **99.7%** | 70.6% | 29.5% |
| fused OR | **100%** | 91.5% | 19.1% |

**Operating point khuyến nghị trình bày:** asymmetric với `--min-precision 0.55` → recall ~97%, FPR ~19%.

---

## 9. Giải trình phương pháp luận

### 9.1. W=3 nhỏ hơn receptive field stem k=7?

Đúng — stem Conv k=7 stride=2 trên W=3 cho output length=2. Đây là **thiết kế có chủ ý**:
- Cửa sổ ngắn bắt **thay đổi nhanh** (transient không bị pha loãng).
- Ablation chứng minh W=3 vượt W=24 **+10 điểm macro-F1**.
- MINA paper cũng dùng receptive field > window cho ECG beat classification.

### 9.2. Label từ CO(GT) percentile — có hợp lệ?

- CO(GT) chỉ dùng **khi train/eval** (có ground truth).
- Deploy path **không dùng CO(GT)** — chỉ MOX + hard-logic.
- Percentile trên train đảm bảo 3 lớp cân bằng (~33% mỗi lớp).
- Hạn chế: UCI ≠ môi trường VN — cần field calibration.

### 9.3. Distribution shift train/test?

| Split | Safe | Warning | Danger |
|-------|------|---------|--------|
| Train | 33% | 33% | 33% |
| Test | 35% | 39% | 26% |

Test lệch về Warning — giải thích Warning F1 thấp hơn Safe/Danger. Dùng **macro-F1** (không weighted) để không bias.

### 9.4. Vì sao ưu tiên Danger Recall?

Hệ thống an toàn công nghiệp (IEC 61508): **chi phí False Negative (bỏ sót Danger) >> False Positive**. Asymmetric policy cho phép chọn operating point phù hợp.

---

## 10. Hạn chế trung thực

| Hạn chế | Hướng xử lý |
|---------|-------------|
| UCI dataset ≠ VN thực tế | Field data collection sau sơ tuyển |
| RF accuracy > MINA trên tabular | Trade-off: deployability vs raw accuracy |
| Safety mode FPR cao | Tune `--min-precision`, kết hợp hard-logic |
| Chưa có Sign-SGD on-chip | Roadmap vòng hoàn thiện |
| Ablation 12 epochs (không 30) | Đủ cho xu hướng; full 30 epochs cho báo cáo cuối |

---

## 11. Q&A — Chuẩn bị trước hội đồng

### Câu 1: "Tại sao không dùng LSTM/Transformer?"

> LSTM cần ~50K+ params, không map được lên MINA accelerator 4K params. Transformer cần attention O(n²) — không phù hợp < 5 mW. MINA 1D-CNN đạt 71% macro-F1 với 4K params, Q1.15, inference < 10 µs.

### Câu 2: "Random Forest accuracy cao hơn — sao cần CNN?"

> RF 76.6% nhưng cần ~200 trees, không lượng tử hóa, không chạy trên NPU. MINA 70.7% với 4K params deployable + safety layer đưa Danger Recall lên 99.7%. Đây là trade-off **Edge AI**, không phải cloud ML.

### Câu 3: "Ablation chứng minh gì?"

> Window W=3 đóng góp **+10 điểm macro-F1** — lớn nhất trong 4 cải tiến. Transient/Focal cần kết hợp W=3 mới phát huy. Multi-task thêm +0.4% và CO MAE 0.48 ppm.

### Câu 4: "Kết quả có tái lập không?"

> 3 seeds: accuracy **69.3% ± 1.8%**, macro-F1 **69.8% ± 1.7%**. Std nhỏ chứng minh pipeline ổn định sau khi thêm AdamW + scheduler + early stopping.

### Câu 5: "Dataset UCI có đại diện không?"

> Thừa nhận hạn chế. UCI là benchmark chuẩn cho air quality ML. Pipeline hoàn chỉnh sẵn sàng cho field data. Hard-logic thresholds calibrate tại chỗ.

### Câu 6: "Q1.15 có mất accuracy không?"

> **0% drop** — đã verify trên test set (`quantize_metrics.json`). Fixed-point Q1.15 phù hợp MINA hardware.

### Câu 7: "Safety 100% recall — có thật không?"

> 100% là **fused OR** (soft AI + hard-logic), không phải pure CNN. Trade-off: FPR 91.5%. Operating point cân bằng: asymmetric + min-precision → 97% recall, 19% FPR.

### Câu 8: "Continuous learning đã làm chưa?"

> Chưa implement on-chip. Roadmap 3 tầng: (1) calibration z-score, (2) policy tuning, (3) Sign-SGD cls_head only. Tầng 1-2 đã có code Python.

---

## 12. Lệnh chạy experiments

```bash
cd ml

# Training (ổn định)
python src/train.py --epochs 30 --window-size 3 --scheduler cosine --patience 8

# Baseline đối chứng
python src/baselines.py

# Ablation study (7 configs, ~15 phút)
python src/ablation.py --epochs 12 --patience 5

# Ablation nhanh (3 configs cuối)
python src/ablation.py --quick --epochs 12

# Đa seed robustness
python src/run_seeds.py --seeds 42 1 7 123 2024 --epochs 20

# Safety eval
python src/evaluate.py --ckpt results/best_model.pt

# Quantize
python src/quantize.py
```

---

## 13. Cấu trúc file mới

```
ml/src/
├── train.py          # + AdamW, scheduler, early stopping, smooth checkpoint
├── baselines.py      # NEW — classical ML baselines
├── ablation.py       # NEW — ablation study
├── run_seeds.py      # NEW — multi-seed robustness
├── evaluate.py       # (giữ nguyên)
└── ...

ml/results/
├── baselines.json
├── ablation.json
├── seed_robustness.json
├── ablation/*.pt
└── seeds/*.pt
```

---

## 14. Slide AI đề xuất (cho bộ slide sơ tuyển)

1. **Bài toán:** Edge AI giám sát khí thải, không cloud
2. **Pipeline:** 9ch → MINA → 3 lớp + Q1.15
3. **Baseline:** Bảng so sánh RF/LR/MLP vs MINA (+ lý do chọn CNN)
4. **Ablation:** Biểu đồ cột macro-F1 theo từng cải tiến
5. **Đa seed:** Mean ± std error bars
6. **Safety:** 3 lớp phòng vệ + operating point
7. **Hardware map:** MINA → accelerator, Q1.15 weights 8KB
8. **Hạn chế & roadmap:** UCI, Sign-SGD, field data

---

*Báo cáo này phục vụ bảo vệ phần AI tại Vòng sơ tuyển. Số liệu từ experiments chạy ngày 27/06/2026.*
