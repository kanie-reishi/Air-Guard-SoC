"""Generate PDF report for AirGuard-AI SoC (Vong so tuyen)."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

DOCS_DIR = Path(__file__).resolve().parent
OUT_PDF = DOCS_DIR / "BAO_CAO_VONG_SO_TUYEN.pdf"

# Windows fonts with Vietnamese support
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\times.ttf"),
]


def find_font() -> Path:
    for p in FONT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("No suitable TTF font found for Vietnamese PDF.")


class ReportPDF(FPDF):
  def __init__(self) -> None:
    super().__init__()
    font = find_font()
    self.add_font("Body", "", str(font))
    self.add_font("Body", "B", str(font))
    self.set_auto_page_break(auto=True, margin=18)
    self.set_left_margin(15)
    self.set_right_margin(15)

  def header(self) -> None:
    self.set_font("Body", "B", 9)
    self.set_text_color(100, 100, 100)
    self.cell(0, 8, "AirGuard-AI SoC — Bao cao Vong so tuyen 2026", align="R", new_x="LMARGIN", new_y="NEXT")
    self.ln(2)

  def footer(self) -> None:
    self.set_y(-15)
    self.set_font("Body", "", 8)
    self.set_text_color(120, 120, 120)
    self.cell(0, 10, f"Trang {self.page_no()}", align="C")

  def cover(self) -> None:
    self.add_page()
    self.set_font("Body", "B", 22)
    self.set_text_color(20, 60, 120)
    self.ln(25)
    self.multi_cell(0, 12, "BAO CAO TINH HINH DU AN\nAIRGUARD-AI SoC", align="C")
    self.ln(8)
    self.set_font("Body", "", 13)
    self.set_text_color(50, 50, 50)
    self.multi_cell(0, 8, "Cuoc thi: Thiet ke Vi mach cho Do thi Thong minh — Lan 3 (2026)\nVong: So tuyen", align="C")
    self.ln(15)
    self.set_font("Body", "", 11)
    info = [
      "Nhom: Nguyen Thanh Chinh, Nguyen Duc Hai, Nguyen Dinh Chinh, Le Van Tam",
      "GVHD: Nguyen Hung Son — BTEC FPT Ha Noi",
      "Ngay cap nhat: 27/06/2026",
    ]
    for line in info:
      self.cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")
    self.ln(20)
    self.set_draw_color(20, 60, 120)
    self.set_line_width(0.8)
    self.line(30, self.get_y(), 180, self.get_y())

  def section(self, title: str) -> None:
    self.ln(4)
    self.set_font("Body", "B", 14)
    self.set_text_color(20, 60, 120)
    self.multi_cell(0, 9, title)
    self.set_draw_color(200, 210, 230)
    self.line(10, self.get_y(), 200, self.get_y())
    self.ln(3)

  def sub(self, title: str) -> None:
    self.ln(2)
    self.set_font("Body", "B", 11)
    self.set_text_color(40, 40, 40)
    self.multi_cell(0, 7, title)
    self.ln(1)

  def body(self, text: str) -> None:
    self.set_x(self.l_margin)
    self.set_font("Body", "", 10)
    self.set_text_color(30, 30, 30)
    self.multi_cell(0, 6, text)
    self.ln(1)

  def bullet(self, text: str) -> None:
    self.set_x(self.l_margin)
    self.set_font("Body", "", 10)
    self.set_text_color(30, 30, 30)
    self.multi_cell(0, 6, f"- {text}")

  def table(self, headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None) -> None:
    if col_widths is None:
      w = int((self.w - self.l_margin - self.r_margin) / len(headers))
      col_widths = [w] * len(headers)
    self.set_font("Body", "B", 9)
    self.set_fill_color(230, 238, 248)
    for i, h in enumerate(headers):
      self.cell(col_widths[i], 8, h, border=1, fill=True)
    self.ln()
    self.set_x(self.l_margin)
    self.set_font("Body", "", 9)
    fill = False
    for row in rows:
      if self.get_y() > 265:
        self.add_page()
      if fill:
        self.set_fill_color(248, 250, 252)
      else:
        self.set_fill_color(255, 255, 255)
      for i, cell in enumerate(row):
        self.cell(col_widths[i], 7, cell[:40], border=1, fill=True)
      self.ln()
      fill = not fill
    self.ln(2)


def build() -> None:
  pdf = ReportPDF()
  pdf.cover()
  pdf.add_page()

  # 1. Tom tat
  pdf.section("1. TOM TAT CHO DONG DOI & GVHD")
  pdf.body(
    "Vong so tuyen yeu cau de xuat y tuong, phuong an va ke hoach — KHONG bat buoc RTL hoan chinh. "
    "Du an DU SUC qua so tuyen neu tap trung vao: (1) y tuong ro rang, (2) minh chung ML da chay, "
    "(3) bang cong cu EDA chi tiet (15 diem), (4) ke hoach RTL/FPGA cu the."
  )
  pdf.sub("Diem manh (da co)")
  pdf.table(
    ["Hang muc", "Trang thai"],
    [
      ["Pipeline ML (train/eval/quantize)", "XONG"],
      ["Model MINA 4.018 params", "XONG"],
      ["Safety 3 lop (asymmetric + OR)", "XONG"],
      ["Q1.15 — 0% accuracy drop", "XONG"],
      ["RTL RV32 (7 module SV)", "DANG LAM"],
    ],
    [110, 80],
  )
  pdf.sub("Diem yeu (can bo sung)")
  pdf.table(
    ["Hang muc", "Uu tien"],
    [
      ["Slide + poster so tuyen", "RAT CAO"],
      ["Bang cong cu EDA (muc 2.3)", "RAT CAO"],
      ["Block diagram SoC thong nhat", "CAO"],
      ["RTL MINA accelerator", "Ke hoach"],
      ["Thong nhat tai lieu NPU/MINA", "CAO"],
    ],
    [110, 80],
  )

  # 2. Bai toan
  pdf.section("2. BAI TOAN & Y TUONG")
  pdf.body(
    "Giam sat khi thai cong nghiep / moi truong do thi can phan hoi nhanh, tiet kiem nang luong, "
    "khong phu thuoc cloud. Cam bien MOX bi drift theo thoi gian. He thong an toan can fail-safe."
  )
  pdf.sub("Giai phap AirGuard-AI SoC")
  pdf.body(
    "Cam bien MOX -> ADC -> Firmware -> MINA NPU (Q1.15) -> Softmax + Asymmetric Policy "
    "-> OR voi Hard Logic (S1 + transient) -> GPIO/UART canh bao."
  )
  pdf.sub("Diem khac biet (tinh moi — 30 diem)")
  for b in [
    "TinyML tren chip: MINA accelerator (~4K params, IEEE TCAS-I 2024)",
    "3 lop phong ve: AI mem + nguong bat doi xung + hard-logic OR",
    "Edge-first: Privacy by design — khong gui raw sensor len cloud",
    "Roadmap Sign-SGD: thich nghi sensor drift (vong sau)",
  ]:
    pdf.bullet(b)

  # 3. Ket qua ML
  pdf.section("3. KET QUA ML DA CO")
  pdf.table(
    ["Metric", "Gia tri"],
    [
      ["Test accuracy", "70.7%"],
      ["Macro-F1", "71.3%"],
      ["Danger Recall (argmax)", "87.6%"],
      ["Parameters", "4,018"],
      ["MACs/inference", "~6,832"],
      ["Q1.15 drop", "0%"],
      ["CO MAE", "0.48 ppm"],
    ],
    [95, 95],
  )
  pdf.sub("Safety-critical")
  pdf.table(
    ["Che do", "Danger Recall", "FPR", "Macro-F1"],
    [
      ["argmax (can bang)", "87.6%", "0.4%", "71.3%"],
      ["asymmetric", "99.7%", "70.6%", "29.5%"],
      ["fused OR", "100%", "91.5%", "19.1%"],
    ],
    [48, 47, 47, 48],
  )
  pdf.body(
    "Khuyen nghi trinh bay: dung operating point asymmetric voi min-precision 0.55 "
    "(recall ~97%, FPR ~19%) thay vi khoe 100% recall."
  )

  # 4. Kien truc
  pdf.section("4. KIEN TRUC PHAN CUNG DU KIEN")
  pdf.table(
    ["Thanh phan", "Mo ta"],
    [
      ["RV32IMF @ 50MHz", "Dieu phoi, firmware, UART/GPIO/WDT"],
      ["MINA @ 100MHz", "PEA + SBA, Weight SRAM Q1.15 (~8KB)"],
      ["Fail-Safe HW", "2 comparator + OR gate -> GPIO"],
      ["CDC", "Async FIFO giua 2 clock domain"],
    ],
    [55, 135],
  )
  pdf.sub("Muc tieu PPA")
  pdf.table(
    ["Thong so", "Muc tieu"],
    [["Cong suat", "< 5 mW"], ["Latency", "< 100 ms"], ["Dien tich", "~0.40 mm2 (65nm)"], ["So hoc", "Q1.15"]],
    [95, 95],
  )

  # 5. Tieu chi
  pdf.section("5. ANH XA TIEU CHI VONG SO TUYEN (100 diem)")
  pdf.table(
    ["Tieu chi", "Diem", "San sang"],
    [
      ["1.1 Tinh moi, sang tao", "30", "CAO"],
      ["1.2 Kha thi, ung dung", "10", "CAO"],
      ["1.4 Low power/area", "10", "TB"],
      ["2.1-2.2 Ky thuat vi mach", "15", "TB"],
      ["2.3 Cong cu EDA", "15", "CAN SLIDE"],
    ],
    [90, 30, 70],
  )

  # 6. Cong cu EDA
  pdf.section("6. CONG CU & THIET BI DU KIEN (MUC 2.3 — 15 DIEM)")
  pdf.table(
    ["Hang muc", "Cong cu"],
    [
      ["HDL", "SystemVerilog / Verilog"],
      ["Mo phong", "ModelSim / Vivado Sim / Verilator"],
      ["Golden model", "Python/PyTorch (DA CO)"],
      ["Synthesis FPGA", "Vivado — Basys 3 Artix-7"],
      ["Synthesis ASIC", "Yosys + OpenLane"],
      ["Physical", "OpenROAD / Innovus (ke hoach)"],
      ["Kiem tra", "DRC/LVS, STA"],
      ["Do luong", "ILA, oscilloscope"],
    ],
    [55, 135],
  )

  # 7. Checklist
  pdf.section("7. VIEC CAN LAM NGAY")
  pdf.sub("Ho so bat buoc")
  for item in [
    "Slide 15-20 trang bam tieu chi 1.1 -> 2.3",
    "Poster (neu yeu cau)",
    "Thuyet minh ky thuat (file PDF nay)",
    "Tai lieu gioi thieu du an 2-3 trang",
  ]:
    pdf.bullet(f"[ ] {item}")
  pdf.sub("Slide quan trong nhat")
  for item in [
    "Bai toan + do thi xanh",
    "Block diagram SoC",
    "Ket qua ML + confusion matrix",
    "Safety 3 lop",
    "BANG CONG CU EDA (15 diem)",
    "Quy trinh verification",
    "Ke hoach Gantt + phan cong",
  ]:
    pdf.bullet(item)
  pdf.sub("Sua tai lieu")
  for item in [
    "Chot MINA accelerator (khong con 16-MAC generic)",
    "Sign-SGD = roadmap vong hoan thien",
    "Cap nhat rv32imf/docs/AirGuardSoC.md",
  ]:
    pdf.bullet(item)

  # 8. Timeline
  pdf.section("8. KE HOACH CAC VONG TIEP THEO")
  pdf.table(
    ["Vong", "Yeu cau", "Viec lam"],
    [
      ["So tuyen", "De xuat + ke hoach", "Slide, poster, ho so"],
      ["Hoan thien", "Mentoring, RTL", "MINA RTL, testbench"],
      ["Ban ket", "RTL + FPGA demo", "Basys 3 bitstream"],
      ["Chung ket", "San pham day du", "Demo end-to-end"],
    ],
    [32, 50, 98],
  )

  # 9. Q&A
  pdf.section("9. CHUAN BI Q&A")
  pdf.sub('Cau 1: "Chung minh phan vi mach o dau?"')
  pdf.body(
    "Golden model Q1.15 + 7 module RTL RV32 + ke hoach MINA (IEEE paper) + Basys 3 FPGA."
  )
  pdf.sub('Cau 2: "Tai sao khong dung cloud?"')
  pdf.body("Privacy, latency <100ms, <5mW — phu hop IoT do thi xanh.")
  pdf.sub('Cau 3: "Dataset UCI co phu hop VN?"')
  pdf.body(
    "Thua nhan han che; da co pipeline hoan chinh; ke hoach thu thap field data sau vong so tuyen."
  )

  pdf.ln(5)
  pdf.set_font("Body", "", 9)
  pdf.set_text_color(100, 100, 100)
  pdf.multi_cell(
    0,
    5,
    "Phu luc: Repo Air-Guard-SoC/ml/ (ML hoan chinh), rv32imf/rtl/ (7 module SV). "
    "Minh chung: ml/results/*.json, *.png, q15_weights/.",
  )

  pdf.output(str(OUT_PDF))
  print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
  build()
