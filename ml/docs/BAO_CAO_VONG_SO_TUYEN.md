# BÁO CÁO TÌNH HÌNH DỰ ÁN AIRGUARD-AI SoC

**Cuộc thi:** Thiết kế Vi mạch cho Đô thị Thông minh — Lần 3 (2026)  
**Vòng:** Sơ tuyển  
**Nhóm:** Hồ Chí Công, Nguyễn Thanh Chính, Nguyễn Đức Hải, Nguyễn Đình Chính, Lê Văn Tâm  
**GVHD:** Nguyễn Hùng Sơn — BTEC FPT Hà Nội  
**Ngày cập nhật:** 27/06/2026

---

## Mục lục

1. [Tóm tắt cho đồng đội & GVHD](#1-tóm-tắt-cho-đồng-đội--gvhd)
2. [Bài toán & ý tưởng](#2-bài-toán--ý-tưởng)
3. [Tiến độ hiện tại](#3-tiến-độ-hiện-tại)
4. [Kết quả ML đã có](#4-kết-quả-ml-đã-có)
5. [Kiến trúc phần cứng dự kiến](#5-kiến-trúc-phần-cứng-dự-kiến)
6. [Ánh xạ tiêu chí Vòng sơ tuyển](#6-ánh-xạ-tiêu-chí-vòng-sơ-tuyển)
7. [Việc cần làm ngay](#7-việc-cần-làm-ngay)
8. [Kế hoạch các vòng tiếp theo](#8-kế-hoạch-các-vòng-tiếp-theo)
9. [Rủi ro & cách xử lý](#9-rủi-ro--cách-xử-lý)

---

## 1. Tóm tắt cho đồng đội & GVHD

### Điểm mạnh (đã có sẵn)

| Hạng mục | Trạng thái | Ghi chú |
|----------|------------|---------|
| Pipeline ML hoàn chỉnh | **Xong** | Train / eval / quantize Q1.15 |
| Model MINA 1D-CNN | **Xong** | 4.018 params, ~6.832 MACs |
| Kết quả số liệu thật | **Xong** | 70.7% acc, 71.3% macro-F1 |
| Lượng tử hóa Q1.15 | **Xong** | 0% accuracy drop |
| Safety-critical (3 lớp) | **Xong** | Asymmetric + OR fusion |
| Đặc tả SoC & MMIO | **Xong** | Trong `ml/docs/AirGuardSoC.md` |
| RTL RISC-V (một phần) | **Đang làm** | 7 module SystemVerilog |

### Điểm yếu (cần bổ sung cho sơ tuyển)

| Hạng mục | Trạng thái | Ưu tiên |
|----------|------------|---------|
| RTL MINA accelerator | **Chưa có** | Cao (trình bày kế hoạch) |
| Mô phỏng RTL + waveform | **Chưa có** | Cao |
| Block diagram SoC thống nhất | **Cần vẽ lại** | Cao |
| Bảng công cụ EDA (mục 2.3 — 15đ) | **Chưa có slide** | **Rất cao** |
| Thống nhất tài liệu (NPU 16-MAC vs MINA) | **Cần sửa** | Cao |
| Sign-SGD / Continuous Learning | **Chỉ ý tưởng** | Trung bình |
| Slide / poster sơ tuyển | **Chưa có** | **Rất cao** |

### Kết luận nhanh

> **Vòng sơ tuyển yêu cầu đề xuất + phương án + kế hoạch — KHÔNG bắt buộc RTL hoàn chỉnh.** Dự án **đủ sức qua sơ tuyển** nếu tập trung vào: (1) trình bày ý tưởng rõ ràng, (2) minh chứng ML đã chạy, (3) bảng công cụ EDA chi tiết (15 điểm), (4) kế hoạch RTL/FPGA cụ thể.

---

## 2. Bài toán & ý tưởng

### Vấn đề thực tế

- Giám sát khí thải công nghiệp / môi trường đô thị cần **phản hồi nhanh**, **tiết kiệm năng lượng**, **không phụ thuộc cloud**.
- Cảm biến MOX bị **drift** theo thời gian → cần thích nghi tại chỗ.
- Hệ thống an toàn công nghiệp cần **fail-safe** — không được bỏ sót nguy hiểm.

### Giải pháp AirGuard-AI SoC

```
Cảm biến MOX → ADC → Firmware preprocess
                              ↓
                    MINA NPU (Edge AI, Q1.15)
                              ↓
              Softmax + Asymmetric Policy
                              ↓
         OR ← Hard Logic (S1 + transient)
                              ↓
                    GPIO / UART → Cảnh báo
```

### Điểm khác biệt (tính mới — tiêu chí 1.1: 30đ)

1. **TinyML trên chip:** MINA accelerator (tham chiếu IEEE TCAS-I 2024), chỉ ~4K params.
2. **3 lớp phòng vệ an toàn:** AI mềm + ngưỡng bất đối xứng + hard-logic OR.
3. **Edge-first:** Privacy by design — không gửi raw sensor lên cloud.
4. **Roadmap Sign-SGD:** Thích nghi sensor drift trên chip (vòng sau).

### Liên quan chủ đề "Đô thị xanh" (tiêu chí 1.4: 10đ)

- Giám sát chất lượng không khí IoT, công suất mục tiêu **< 5 mW**.
- Ứng dụng: khu công nghiệp, trạm quan trắc môi trường, smart city sensor node.
- Giải quyết **low power + low area + edge AI** — trực tiếp chủ đề cuộc thi.

---

## 3. Tiến độ hiện tại

### 3.1. Phần mềm / ML (`ml/`)

| File / Module | Mô tả | Trạng thái |
|---------------|-------|------------|
| `dataset.py` | 9 kênh + transient, multi-task CO reg | Xong |
| `model.py` | MinaMultiTask1D (Inception + SE + dual head) | Xong |
| `losses.py` | Focal Loss + preset safety | Xong |
| `train.py` | Multi-task training, CLI đầy đủ | Xong |
| `evaluate.py` | Safety eval: argmax / asymmetric / fused | Xong |
| `inference_policy.py` | Asymmetric threshold + sweep | Xong |
| `safety_fusion.py` | Hard-logic OR fusion | Xong |
| `quantize.py` | Export Q1.15 weights | Xong |
| `sweep_window.py` | W=3h vs W=6h | Xong |
| `results/best_model.pt` | Checkpoint tốt nhất | Xong |
| `results/q15_weights/` | Trọng số deploy | Xong |

### 3.2. Phần cứng (`rv32imf/`)

| Hạng mục | Trạng thái |
|----------|------------|
| `rtl/alu.sv`, `decoder.sv`, `branch_unit.sv`, ... | Có (7 file) — lõi RV32IMF |
| RTL MINA accelerator | Chưa |
| RTL NPU / MAC array | Chưa |
| RTL fail-safe (comparator + OR) | Chưa |
| Testbench | Chưa |
| FPGA synthesis | Chưa |

### 3.3. Tài liệu

| Tài liệu | Ghi chú |
|----------|---------|
| `ml/README.md` | Pipeline + kết quả + safety |
| `ml/docs/AirGuardSoC.md` | Báo cáo kỹ thuật đầy đủ |
| `ml/docs/MINA.md` | Paper MINA tham chiếu |
| `ml/docs/THE_LE_CUOC_THI.md` | Thể lệ đầy đủ |
| `rv32imf/docs/AirGuardSoC.md` | Spec gốc (encoding lỗi, cần cập nhật) |

---

## 4. Kết quả ML đã có

### 4.1. Phân loại (checkpoint `best_model.pt`)

| Metric | Giá trị |
|--------|---------|
| Test accuracy | **70.7%** |
| Test macro-F1 | **71.3%** |
| Danger Recall (argmax) | **87.6%** |
| Danger Precision | 59.8% |
| Warning F1 | ~0.61 |
| CO MAE (regression) | **0.48 ppm** |
| Parameters | **4.018** |
| MACs / inference | **~6.832** |
| Q1.15 accuracy drop | **0%** |
| Window size | **3 giờ** |
| Input | **9 kênh × 3 timestep** |

*Dataset: UCI Air Quality (De Vito), split temporal 70/15/15.*

### 4.2. Safety-critical (3 chế độ)

| Chế độ | Danger Recall | Safe→Danger FPR | Macro-F1 |
|--------|---------------|-----------------|----------|
| argmax (cân bằng) | 87.6% | **0.4%** | **71.3%** |
| asymmetric (an toàn) | **99.7%** | 70.6% | 29.5% |
| fused OR (tối đa) | **100%** | 91.5% | 19.1% |

**Khuyến nghị trình bày:** Dùng operating point asymmetric với `--min-precision 0.55` → recall ~97%, FPR ~19% (cân bằng hơn).

### 4.3. Minh chứng "mô phỏng ban đầu" (thể lệ khuyến khích)

Các file có thể đính kèm hồ sơ:

- `ml/results/eval_metrics.json`
- `ml/results/safety_metrics_baseline.json`
- `ml/results/quantize_metrics.json`
- `ml/results/danger_pr_sweep.png`
- `ml/results/confusion_matrix_fused.png`
- `ml/results/q15_weights/q15_manifest.json`

---

## 5. Kiến trúc phần cứng dự kiến

### 5.1. SoC tổng thể (phiên bản thống nhất — dùng cho slide)

```
┌─────────────────────────────────────────────────────────┐
│                    AirGuard-AI SoC                       │
│  ┌──────────────────┐    ┌──────────────────────────┐ │
│  │ System 50 MHz    │    │ Compute 100 MHz          │ │
│  │ RV32IMF          │AXI │ MINA Accelerator         │ │
│  │ UART/GPIO/WDT    │◄──►│ PEA + SBA + Weight SRAM  │ │
│  │ Firmware:        │    │ Q1.15 (~8 KB)            │ │
│  │  preprocess      │    └──────────────────────────┘ │
│  │  softmax/policy  │                                  │
│  │  Sign-SGD (sau)  │    ┌──────────────────────────┐ │
│  └──────────────────┘    │ Fail-Safe HW             │ │
│                            │ 2× Comparator + OR → GPIO│ │
│  ADC ← MOX Sensors         └──────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 5.2. Mục tiêu PPA

| Thông số | Mục tiêu | Ghi chú |
|----------|----------|---------|
| Công suất | < 5 mW | Clock gating, dual clock domain |
| Latency | < 100 ms | Inference ước tính < 1 ms |
| Diện tích | ~0.40 mm² | ASIC 65nm (ước lượng) |
| Số học | Q1.15 fixed-point | Đã verify trên ML |

### 5.3. Map model → phần cứng

| Thành phần ML | Hiện thực |
|---------------|-----------|
| Stem + Inception ×2 + Residual | MINA PEA (RTL) |
| SE Attention | RV32 hoặc PE nhỏ |
| Cls / Reg head | RV32 (~120 MACs) |
| Softmax + asymmetric | Firmware C |
| Hard-logic S1 + transient | Comparator RTL |

### 5.4. Công cụ & thiết bị dự kiến (mục 2.3 — 15 điểm)

| Hạng mục | Công cụ / thiết bị |
|----------|-------------------|
| HDL | SystemVerilog / Verilog |
| Mô phỏng | ModelSim / Vivado Simulator / Verilator |
| Golden model | Python/PyTorch (đã có) |
| Synthesis FPGA | Vivado 2024 (Basys 3, Artix-7) |
| Synthesis ASIC | Yosys + OpenLane (Sky130) hoặc Design Compiler |
| Physical design | OpenROAD / Innovus (kế hoạch) |
| Kiểm tra | DRC/LVS (Magic/KLayout), STA (OpenSTA) |
| Debug | ILA (Integrated Logic Analyzer) |
| Đo lường | Oscilloscope, power meter |

---

## 6. Ánh xạ tiêu chí Vòng sơ tuyển

**Tổng điểm tối đa: 100** (Ý tưởng 50 + Kỹ thuật 40 + 1.3 = 0)

| Tiêu chí | Điểm | Nội dung trình bày | Mức sẵn sàng |
|----------|------|-------------------|--------------|
| 1.1 Tính mới, sáng tạo | 30 | Edge AI + safety 3 lớp + MINA | **Cao** |
| 1.2 Khả thi, ứng dụng | 10 | ML proven, IoT air quality | **Cao** |
| 1.4 Low power/area/speed | 10 | 4K params, <5mW, <100ms | **Trung bình** (ước lượng) |
| 2.1 Thiết kế/mô phỏng/kiểm tra | 5 | Quy trình + golden model | **Trung bình** |
| 2.1.3 Kiểm tra thiết kế | 5 | Testbench plan, bit-true | **Trung bình** |
| 2.2 Kỹ thuật vi mạch | 10 | Kiến trúc + kế hoạch RTL/PD | **Trung bình** |
| 2.3 Công cụ/công nghệ | **15** | Bảng EDA + Basys 3 | **Cần làm slide** |

---

## 7. Việc cần làm ngay

### Checklist trước Vòng sơ tuyển

#### A. Hồ sơ bắt buộc

- [ ] **Slide trình bày** (15–20 slide) bám tiêu chí 1.1 → 2.3
- [ ] **Poster** (nếu yêu cầu)
- [ ] **Thuyết minh kỹ thuật sơ bộ** (1 file PDF — bản này)
- [ ] **Tài liệu giới thiệu dự án** (2–3 trang executive summary)

#### B. Nội dung slide quan trọng nhất

- [ ] Slide 1: Bài toán + đô thị xanh
- [ ] Slide 2: Giải pháp SoC (block diagram)
- [ ] Slide 3: Kết quả ML (bảng số liệu + confusion matrix)
- [ ] Slide 4: Safety 3 lớp (sơ đồ OR fusion)
- [ ] Slide 5: **Bảng công cụ EDA** (mục 2.3 — 15đ)
- [ ] Slide 6: Kiến trúc MINA + map layer
- [ ] Slide 7: Quy trình verification (golden model ↔ RTL)
- [ ] Slide 8: Kế hoạch Gantt (sơ tuyển → chung kết)
- [ ] Slide 9: Phân công nhóm
- [ ] Slide 10: Rủi ro & giảm thiểu

#### C. Sửa tài liệu

- [ ] Thống nhất: dùng **MINA accelerator** (không còn "16 MAC generic")
- [ ] Sign-SGD ghi rõ **"roadmap vòng hoàn thiện"** (chưa implement)
- [ ] Sửa encoding `rv32imf/docs/AirGuardSoC.md` hoặc thay bằng `ml/docs/AirGuardSoC.md`

#### D. Minh chứng đính kèm (khuyến khích)

- [ ] Copy `results/*.png` vào folder `docs/minh_chung/`
- [ ] Link repo / zip mã nguồn `ml/`
- [ ] (Tùy chọn) Video 2–3 phút demo chạy `evaluate.py`

### Phân công gợi ý

| Thành viên | Nhiệm vụ sơ tuyển |
|------------|-------------------|
| **ML / Python** | Slide kết quả, demo script, minh chứng JSON/PNG |
| **RTL** | Block diagram, quy trình sim, liệt kê module RTL có |
| **SoC / Firmware** | MMIO map, luồng runtime, pseudocode C |
| **Tài liệu** | Slide tổng hợp, poster, hồ sơ PDF |
| **Tất cả** | Q&A rehearsal — 3 câu hỏi khó ở mục 9 |

---

## 8. Kế hoạch các vòng tiếp theo

| Vòng | Yêu cầu chính | Việc cần làm |
|------|---------------|--------------|
| **Sơ tuyển** (hiện tại) | Ý tưởng + phương án + kế hoạch | Slide, poster, hồ sơ |
| **Hoàn thiện** | Mentoring, RTL, sim data | MINA RTL, testbench, golden model |
| **Bán kết** | RTL + waveform + FPGA demo | Basys 3 bitstream, demo sensor |
| **Chung kết** | Sản phẩm hoàn chỉnh + video | End-to-end demo, báo cáo đầy đủ |

### Timeline gợi ý (sau qua sơ tuyển)

```
Tháng 7/2026  : RTL MINA conv layer + testbench
Tháng 8/2026  : Tích hợp RV32 + MINA trên FPGA
Tháng 9/2026  : Firmware preprocess + policy C
Tháng 10/2026 : Demo Basys 3 + CL benchmark
Tháng 11/2026 : Hoàn thiện hồ sơ bán kết
```

---

## 9. Rủi ro & cách xử lý

### Rủi ro kỹ thuật

| Rủi ro | Cách trả lời khi Q&A |
|--------|----------------------|
| Dataset UCI ≠ VN thực tế | Đã nêu hạn chế; kế hoạch thu thập field data |
| Chưa có RTL MINA | Vòng sơ tuyển chỉ cần kế hoạch; đã có golden model Python |
| Sign-SGD chưa làm | Roadmap rõ 3 tầng: calibration → policy → weight update |
| FPR cao ở safety mode | Operating point có `--min-precision`; trade-off có chủ đích |
| Mâu thuẫn tài liệu NPU/MINA | Đã chốt MINA; spec cũ sẽ cập nhật |

### 3 câu hỏi khó — chuẩn bị trả lời

1. **"Em chứng minh phần vi mạch ở đâu?"**  
   → Golden model Q1.15 + 7 module RTL RV32 + kế hoạch MINA có tham chiếu IEEE paper + Basys 3.

2. **"Tại sao không dùng cloud?"**  
   → Privacy, latency <100ms, <5mW — phù hợp IoT đô thị xanh.

3. **"Khác gì so với sensor thương mại?"**  
   → 3 lớp phòng vệ (AI + threshold + hard-logic), thích nghi drift (roadmap), tự thiết kế SoC.

---

## Phụ lục: Cấu trúc repo

```
Air-Guard-SoC/
├── ml/                    # Pipeline ML (HOÀN CHỈNH)
│   ├── src/               # train, eval, model, safety...
│   ├── results/           # checkpoints, metrics, q15_weights
│   └── docs/              # tài liệu + thể lệ + báo cáo này
└── rv32imf/               # Phần cứng (ĐANG LÀM)
    ├── rtl/               # 7 module SystemVerilog
    └── docs/              # spec SoC
```

---

*Báo cáo này phục vụ nội bộ nhóm và GVHD, chuẩn bị Vòng sơ tuyển Cuộc thi Thiết kế Vi mạch cho Đô thị Thông minh lần 3.*
