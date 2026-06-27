"""Generate PDF for BAO_VE_AI_PIPELINE defense document."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

DOCS_DIR = Path(__file__).resolve().parent
OUT_PDF = DOCS_DIR / "BAO_VE_AI_PIPELINE.pdf"

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
]


def find_font() -> Path:
    for p in FONT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("No TTF font found.")


class DocPDF(FPDF):
    def __init__(self) -> None:
        super().__init__()
        font = find_font()
        self.add_font("Body", "", str(font))
        self.add_font("Body", "B", str(font))
        self.set_auto_page_break(auto=True, margin=18)
        self.set_left_margin(15)
        self.set_right_margin(15)

    def section(self, title: str) -> None:
        self.ln(3)
        self.set_font("Body", "B", 13)
        self.set_text_color(20, 60, 120)
        self.multi_cell(0, 8, title)
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
        self.multi_cell(0, 6, f"- {text}")

    def table(self, headers: list[str], rows: list[list[str]], widths: list[int]) -> None:
        self.set_font("Body", "B", 9)
        self.set_fill_color(230, 238, 248)
        for i, h in enumerate(headers):
            self.cell(widths[i], 8, h, border=1, fill=True)
        self.ln()
        self.set_font("Body", "", 8)
        fill = False
        for row in rows:
            if self.get_y() > 265:
                self.add_page()
            self.set_fill_color(248, 250, 252) if fill else self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(widths[i], 7, cell[:35], border=1, fill=True)
            self.ln()
            fill = not fill
        self.ln(2)


def build() -> None:
    pdf = DocPDF()
    pdf.add_page()
    pdf.set_font("Body", "B", 18)
    pdf.set_text_color(20, 60, 120)
    pdf.multi_cell(0, 10, "BAO VE Y TUONG AI\nAirGuard-AI Pipeline", align="C")
    pdf.ln(5)
    pdf.set_font("Body", "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 7, "Vong so tuyen - Cuoc thi Thiet ke Vi mach Do thi Thong minh 2026", align="C")

    pdf.section("1. Tom tat")
    pdf.body(
        "TinyML tren chip: MINA 1D-CNN, 4018 params, 6832 MACs, Q1.15 (0% drop). "
        "Co baseline, ablation, da seed, training on dinh (AdamW + scheduler + early stop)."
    )

    pdf.section("2. Baseline doi chung (test)")
    pdf.table(
        ["Model", "Acc", "F1", "Danger R"],
        [
            ["Logistic Reg", "74.4%", "75.0%", "89.8%"],
            ["Random Forest", "76.6%", "76.9%", "86.2%"],
            ["MLP", "69.5%", "69.6%", "90.1%"],
            ["MINA CNN", "70.7%", "71.3%", "87.6%"],
        ],
        [55, 30, 30, 35],
    )
    pdf.body(
        "RF accuracy cao hon nhung khong deploy duoc len chip. MINA: 4K params, Q1.15, "
        "safety layer dua Danger Recall len 99.7%."
    )

    pdf.section("3. Ablation study (test)")
    pdf.table(
        ["Config", "Acc", "F1", "MACs"],
        [
            ["8ch W=24 CE", "61.1%", "60.7%", "35520"],
            ["+ transient", "59.0%", "58.3%", "36192"],
            ["+ focal", "57.1%", "57.0%", "36192"],
            ["+ W=3", "69.6%", "70.3%", "6832"],
            ["+ multi-task", "70.3%", "70.7%", "6832"],
            ["W=6 control", "71.8%", "72.0%", "9768"],
        ],
        [50, 28, 28, 30],
    )
    pdf.body("W=3 la cai tien lon nhat: +10 diem macro-F1, MACs giam 5x.")

    pdf.section("4. Da seed (n=3)")
    pdf.table(
        ["Metric", "Mean +/- Std"],
        [
            ["Accuracy", "69.3% +/- 1.8%"],
            ["Macro-F1", "69.8% +/- 1.7%"],
            ["Danger Recall", "89.4% +/- 1.7%"],
        ],
        [80, 80],
    )

    pdf.section("5. Safety 3 lop")
    pdf.table(
        ["Mode", "Danger R", "FPR"],
        [
            ["argmax", "87.6%", "0.4%"],
            ["asymmetric", "99.7%", "70.6%"],
            ["fused OR", "100%", "91.5%"],
        ],
        [55, 40, 40],
    )

    pdf.section("6. Lenh chay")
    for cmd in [
        "python src/baselines.py",
        "python src/ablation.py --epochs 12",
        "python src/run_seeds.py --seeds 42 1 7",
        "python src/train.py --scheduler cosine --patience 8",
    ]:
        pdf.bullet(cmd)

    pdf.section("7. Q&A nhanh")
    qa = [
        ("Tai sao CNN khong LSTM?", "4K params, map MINA HW, <10us inference."),
        ("RF cao hon?", "Trade-off deployability vs raw accuracy."),
        ("Tai lap?", "3 seeds, std ~1.7%."),
        ("Q1.15?", "0% accuracy drop verified."),
    ]
    for q, a in qa:
        pdf.set_font("Body", "B", 10)
        pdf.body(f"Q: {q}")
        pdf.set_font("Body", "", 10)
        pdf.body(f"A: {a}")

    pdf.output(str(OUT_PDF))
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    build()
