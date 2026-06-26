# RV32IMF Processor Specification
**Version:** 1.0  
**HDL:** SystemVerilog  
**Target:** Xilinx/AMD FPGA (Vivado) + Simulation  
**Date:** 2026-04-25

---

## Table of Contents
1. [Overview](#1-overview)
2. [ISA Specification](#2-isa-specification)
3. [Microarchitecture Overview](#3-microarchitecture-overview)
4. [Pipeline Stages](#4-pipeline-stages)
5. [Register Files](#5-register-files)
6. [Hazard Handling](#6-hazard-handling)
7. [Floating Point Unit (FPU)](#7-floating-point-unit-fpu)
8. [Memory Architecture](#8-memory-architecture)
9. [Exception & Interrupt Handling](#9-exception--interrupt-handling)
10. [CSR Map](#10-csr-map)
11. [Module Hierarchy](#11-module-hierarchy)
12. [Top-Level Interface](#12-top-level-interface)
13. [Memory Map](#13-memory-map)
14. [Verification Plan](#14-verification-plan)
15. [File Structure](#15-file-structure)

---

## 1. Overview

This document specifies a **32-bit RISC-V processor** implementing the **RV32IMF** ISA:

| Extension | Description |
|-----------|-------------|
| **RV32I**  | Base 32-bit integer instruction set |
| **M**      | Integer multiply and divide |
| **F**      | Single-precision IEEE 754 floating-point |

The processor uses a **classic 5-stage in-order pipeline** (IF → ID → EX → MEM → WB) with:
- Full integer forwarding/bypassing to minimize stalls
- A **multi-cycle FPU** attached to the EX stage with stall-based integration
- **Harvard memory architecture** (separate instruction and data memories)
- **Machine-mode (M-mode) only** exception and trap support
- Target: **Xilinx/AMD FPGA** synthesized with Vivado; also fully simulatable

---

## 2. ISA Specification

### 2.1 RV32I Base Instructions
All 47 base integer instructions are implemented:

| Category | Instructions |
|----------|-------------|
| Integer arithmetic | `ADD`, `SUB`, `ADDI`, `LUI`, `AUIPC` |
| Logical | `AND`, `OR`, `XOR`, `ANDI`, `ORI`, `XORI` |
| Shift | `SLL`, `SRL`, `SRA`, `SLLI`, `SRLI`, `SRAI` |
| Compare | `SLT`, `SLTU`, `SLTI`, `SLTIU` |
| Branch | `BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU` |
| Jump | `JAL`, `JALR` |
| Load | `LB`, `LH`, `LW`, `LBU`, `LHU` |
| Store | `SB`, `SH`, `SW` |
| System | `ECALL`, `EBREAK`, `FENCE` (NOP), `MRET` |

### 2.2 M Extension (Multiply/Divide)
| Instruction | Operation |
|-------------|-----------|
| `MUL`       | Lower 32 bits of signed × signed |
| `MULH`      | Upper 32 bits of signed × signed |
| `MULHSU`    | Upper 32 bits of signed × unsigned |
| `MULHU`     | Upper 32 bits of unsigned × unsigned |
| `DIV`       | Signed integer divide |
| `DIVU`      | Unsigned integer divide |
| `REM`       | Signed remainder |
| `REMU`      | Unsigned remainder |

MUL/MULH instructions shall complete in **3 cycles** (pipelined DSP).  
DIV/REM instructions shall complete in **up to 34 cycles** (iterative non-restoring divider).

### 2.3 F Extension (Single-Precision Float)
Implements the full RV32F instruction set per RISC-V ISA spec v2.2:

| Category | Instructions |
|----------|-------------|
| Load/Store | `FLW`, `FSW` |
| Arithmetic | `FADD.S`, `FSUB.S`, `FMUL.S`, `FDIV.S`, `FSQRT.S` |
| Fused | `FMADD.S`, `FMSUB.S`, `FNMADD.S`, `FNMSUB.S` |
| Compare | `FEQ.S`, `FLT.S`, `FLE.S` |
| Convert | `FCVT.W.S`, `FCVT.WU.S`, `FCVT.S.W`, `FCVT.S.WU` |
| Move | `FMV.X.W`, `FMV.W.X` |
| Sign | `FSGNJ.S`, `FSGNJN.S`, `FSGNJX.S` |
| Classify | `FCLASS.S` |
| Min/Max | `FMIN.S`, `FMAX.S` |

**Rounding modes:** All 5 IEEE 754 rounding modes supported (RNE, RTZ, RDN, RUP, RMM) via `frm` field in `fcsr`.  
**Exception flags:** NV, DZ, OF, UF, NX tracked in `fflags` within `fcsr`.  
**Special values:** NaN, ±Inf, ±0, denormals handled per IEEE 754-2008.

---

## 3. Microarchitecture Overview

```
        ┌─────────────────────────────────────────────────────────────────┐
        │                     RV32IMF Core                                │
        │                                                                 │
 IMEM ──┤  IF Stage  →  ID Stage  →  EX Stage  →  MEM Stage  →  WB Stage │── DMEM
        │               ↑                                                 │
        │           Register Files                                        │
        │           (INT x32 + FP f32)                                    │
        │                                                                 │
        │           Hazard Unit (Forwarding + Stall + Flush)              │
        │                                                                 │
        │           Multi-Cycle FPU (attached to EX)                     │
        │                                                                 │
        │           CSR File (M-mode: mstatus, mepc, mcause, etc.)       │
        └─────────────────────────────────────────────────────────────────┘
```

---

## 4. Pipeline Stages

### 4.1 IF — Instruction Fetch
- Drives the **Program Counter (PC)** to instruction memory
- Default: `PC ← PC + 4`
- On branch/jump taken: `PC ← branch_target` (resolved in EX)
- On exception/trap: `PC ← mtvec`
- On MRET: `PC ← mepc`
- Inserts a **bubble (NOP)** on flush
- Pipeline register: **IF/ID**

### 4.2 ID — Instruction Decode
- Decodes opcode, funct3, funct7, rs1, rs2, rs3 (for F), rd
- Reads **integer register file** (x0–x31) and **float register file** (f0–f31)
- Generates all control signals: `alu_op`, `mem_we`, `reg_we`, `fpu_op`, `branch_type`, etc.
- Generates immediate: I/S/B/U/J types per RV32 spec
- Detects FPU instruction → asserts `fpu_valid`
- Pipeline register: **ID/EX**

### 4.3 EX — Execute
- **Integer ALU**: ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU, pass-through
- **Branch comparator**: evaluates BEQ/BNE/BLT/BGE/BLTU/BGEU; computes branch target
- **Jump logic**: JAL/JALR target computation, `rd ← PC+4`
- **Multiplier**: 3-cycle pipelined DSP multiply unit (RV32M)
- **Divider**: iterative divider, issues stall until done (up to 34 cycles)
- **FPU interface**: dispatches FP operations; stalls pipeline until `fpu_done`
- **Forwarding MUXes**: select between register file, EX/MEM forward, MEM/WB forward
- Pipeline register: **EX/MEM**

### 4.4 MEM — Memory Access
- Drives **data memory** interface
- Loads: `LB`, `LH`, `LW`, `LBU`, `LHU` — sign/zero extends result
- Stores: `SB`, `SH`, `SW` — byte-enable generation
- FLW/FSW: float load/store, routes data to/from float register file
- Pass-through for non-memory instructions
- Pipeline register: **MEM/WB**

### 4.5 WB — Write Back
- Selects write-back data: ALU result, memory load data, FPU result, PC+4 (for JAL/JALR)
- Writes to **integer register file** or **float register file** based on instruction type
- x0 writes are suppressed (hardwired zero)

---

## 5. Register Files

### 5.1 Integer Register File
- 32 × 32-bit registers (`x0`–`x31`)
- `x0` hardwired to 0 (writes ignored)
- 2 read ports, 1 write port
- Synchronous write, asynchronous read (for forwarding compatibility)

### 5.2 Float Register File
- 32 × 32-bit registers (`f0`–`f31`)
- 3 read ports (for fused multiply-add: rs1, rs2, rs3), 1 write port
- Synchronous write, asynchronous read
- No hardwired-zero register

### 5.3 CSR File
- Machine-mode CSRs only (see Section 10)
- Accessed via CSRRW/CSRRS/CSRRC instructions (implemented as special ALU ops)

---

## 6. Hazard Handling

### 6.1 Data Hazards — Integer Forwarding
Full bypass network for integer results:

| Hazard Type | Forward From | Forward To |
|-------------|-------------|-----------|
| EX-EX       | EX/MEM.alu_result | EX stage ALU input |
| MEM-EX      | MEM/WB.alu_result or load_data | EX stage ALU input |

**Load-use hazard** (LW followed immediately by dependent instruction):
- 1-cycle stall inserted by Hazard Unit
- IF and ID stages frozen; bubble inserted into EX

### 6.2 Data Hazards — FPU
- When an FPU instruction is in-flight, the pipeline **stalls** (all upstream stages frozen)
- FPU result is forwarded to EX stage inputs after `fpu_done` is asserted
- FP load-use: same 1-cycle stall as integer load-use

### 6.3 Control Hazards — Branches
- Branches resolved in **EX stage** (2-cycle penalty)
- On taken branch: **flush IF and ID** pipeline registers (insert 2 bubbles)
- Static **not-taken prediction** (no branch predictor in v1.0)

### 6.4 Structural Hazards
- Harvard memory eliminates IF/MEM structural hazards
- Divider and FPU operations stall the full pipeline until completion

### 6.5 Hazard Unit Signals

| Signal | Direction | Description |
|--------|-----------|-------------|
| `stall_if`  | out | Freeze IF stage |
| `stall_id`  | out | Freeze ID stage |
| `stall_ex`  | out | Freeze EX stage |
| `flush_if`  | out | Insert bubble into IF/ID register |
| `flush_id`  | out | Insert bubble into ID/EX register |
| `flush_ex`  | out | Insert bubble into EX/MEM register |
| `fwd_a_sel` | out | Forward MUX select for ALU input A |
| `fwd_b_sel` | out | Forward MUX select for ALU input B |

---

## 7. Floating Point Unit (FPU)

### 7.1 Architecture
The FPU is a **multi-cycle, non-pipelined unit** attached to the EX stage. While an FP operation is executing, the entire pipeline is stalled. This minimizes area at the cost of throughput.

### 7.2 FPU Interface

| Signal | Width | Direction | Description |
|--------|-------|-----------|-------------|
| `fpu_valid`   | 1  | Core → FPU | Start operation |
| `fpu_op`      | 5  | Core → FPU | Operation code |
| `fpu_rm`      | 3  | Core → FPU | Rounding mode |
| `fpu_src_a`   | 32 | Core → FPU | Operand A |
| `fpu_src_b`   | 32 | Core → FPU | Operand B |
| `fpu_src_c`   | 32 | Core → FPU | Operand C (FMA) |
| `fpu_done`    | 1  | FPU → Core | Result ready |
| `fpu_result`  | 32 | FPU → Core | Float result |
| `fpu_flags`   | 5  | FPU → Core | IEEE exception flags (NV/DZ/OF/UF/NX) |

### 7.3 FPU Sub-Modules

| Sub-module | Operations | Latency |
|-----------|-----------|---------|
| `fpu_add`  | FADD.S, FSUB.S | 3 cycles |
| `fpu_mul`  | FMUL.S | 3 cycles |
| `fpu_div`  | FDIV.S | 12 cycles |
| `fpu_sqrt` | FSQRT.S | 12 cycles |
| `fpu_fma`  | FMADD.S, FMSUB.S, FNMADD.S, FNMSUB.S | 5 cycles |
| `fpu_cvt`  | FCVT.W.S, FCVT.WU.S, FCVT.S.W, FCVT.S.WU | 2 cycles |
| `fpu_misc` | FMIN/FMAX, FEQ/FLT/FLE, FSGNJ, FCLASS, FMV | 1 cycle |

### 7.4 IEEE 754 Compliance
- All results rounded to nearest even (RNE) by default; `frm` overrides per instruction
- Denormal inputs treated as zero (flush-to-zero mode, FPGA area optimization) — **configurable via parameter**
- NaN canonicalization: all NaN outputs are canonical quiet NaN (`0x7FC00000`)

---

## 8. Memory Architecture

### 8.1 Harvard Split Memory

```
         ┌──────────────┐        ┌──────────────┐
         │  Instruction │        │     Data     │
         │    Memory    │        │    Memory    │
         │  (IMEM BRAM) │        │  (DMEM BRAM) │
         └──────┬───────┘        └──────┬───────┘
                │                       │
           IF Stage                MEM Stage
         (read only)           (read + write)
```

### 8.2 Instruction Memory (IMEM)
- Width: 32 bits (one instruction per address)
- Depth: Configurable via `IMEM_DEPTH` parameter (default: 4096 words = 16 KB)
- Interface: Single read port, word-aligned, 1-cycle latency
- FPGA: Mapped to Xilinx Block RAM (Simple Dual Port)
- Initialized from `.mem` hex file at synthesis/simulation

### 8.3 Data Memory (DMEM)
- Width: 32 bits
- Depth: Configurable via `DMEM_DEPTH` parameter (default: 4096 words = 16 KB)
- Interface: Single read/write port, byte-enable (`be[3:0]`) for sub-word stores
- FPGA: Mapped to Xilinx Block RAM (True Dual Port)
- 1-cycle read latency; write-first mode

### 8.4 Memory Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMEM_DEPTH` | 4096 | Instruction memory words |
| `DMEM_DEPTH` | 4096 | Data memory words |
| `IMEM_INIT_FILE` | `""` | Path to hex init file |
| `DMEM_INIT_FILE` | `""` | Path to hex init file |

---

## 9. Exception & Interrupt Handling

### 9.1 Supported Exceptions (M-mode)

| mcause code | Exception |
|-------------|-----------|
| 0 | Instruction address misaligned |
| 2 | Illegal instruction |
| 3 | Breakpoint (`EBREAK`) |
| 4 | Load address misaligned |
| 6 | Store address misaligned |
| 8 | Environment call from M-mode (`ECALL`) |
| 11 | FPU invalid operation (NV flag, configurable) |

### 9.2 Interrupt Support

| mcause code | Interrupt |
|-------------|-----------|
| 3 (bit 31 set) | Machine software interrupt |
| 7 (bit 31 set) | Machine timer interrupt |
| 11 (bit 31 set) | Machine external interrupt |

### 9.3 Trap Flow
1. Exception/interrupt detected → pipeline **flushed** (all stages cleared)
2. `mepc ← PC` of offending/interrupted instruction
3. `mcause ← exception code`
4. `mtval ← faulting address or instruction` (where applicable)
5. `mstatus.MIE → mstatus.MPIE`; `mstatus.MIE ← 0`
6. `PC ← mtvec` (direct mode: `mtvec[1:0] == 2'b00`)
7. Return: `MRET` → `PC ← mepc`; `mstatus.MPIE → mstatus.MIE`

---

## 10. CSR Map

| Address | Name | Description |
|---------|------|-------------|
| `0x001` | `fflags` | FP accrued exception flags |
| `0x002` | `frm` | FP dynamic rounding mode |
| `0x003` | `fcsr` | FP control and status (fflags + frm) |
| `0x300` | `mstatus` | Machine status register |
| `0x301` | `misa` | ISA and extensions (read-only: RV32IMF) |
| `0x304` | `mie` | Machine interrupt enable |
| `0x305` | `mtvec` | Trap vector base address |
| `0x340` | `mscratch` | Scratch register |
| `0x341` | `mepc` | Exception program counter |
| `0x342` | `mcause` | Trap cause |
| `0x343` | `mtval` | Trap value |
| `0x344` | `mip` | Machine interrupt pending |
| `0xF11` | `mvendorid` | Vendor ID (read-only: 0x0) |
| `0xF12` | `marchid` | Architecture ID (read-only: 0x0) |
| `0xF14` | `mhartid` | Hardware thread ID (read-only: 0x0) |
| `0xB00` | `mcycle` | Cycle counter low 32 bits |
| `0xB80` | `mcycleh` | Cycle counter high 32 bits |
| `0xB02` | `minstret` | Instructions retired low 32 bits |
| `0xB82` | `minstreth` | Instructions retired high 32 bits |

---

## 11. Module Hierarchy

```
rv32imf_top                         ← Top-level (FPGA wrapper)
├── rv32imf_core                    ← CPU core
│   ├── if_stage                    ← Instruction Fetch
│   │   └── pc_reg                  ← Program Counter register
│   ├── id_stage                    ← Instruction Decode
│   │   ├── decoder                 ← Opcode decoder + control signals
│   │   ├── imm_gen                 ← Immediate generator
│   │   ├── int_regfile             ← Integer register file (32×32)
│   │   └── fp_regfile              ← Float register file (32×32)
│   ├── ex_stage                    ← Execute
│   │   ├── alu                     ← Integer ALU
│   │   ├── branch_unit             ← Branch comparator + target
│   │   ├── mul_unit                ← 3-cycle pipelined multiplier
│   │   ├── div_unit                ← Iterative divider (up to 34 cycles)
│   │   └── fpu_top                 ← Multi-cycle FPU
│   │       ├── fpu_add             ← Adder/Subtractor
│   │       ├── fpu_mul             ← Multiplier
│   │       ├── fpu_fma             ← Fused Multiply-Add
│   │       ├── fpu_div             ← Divider
│   │       ├── fpu_sqrt            ← Square Root
│   │       ├── fpu_cvt             ← Int↔Float converter
│   │       └── fpu_misc            ← Compare, Sign, Move, Classify
│   ├── mem_stage                   ← Memory Access
│   │   └── load_store_unit         ← Byte/Half/Word align + byte-enable
│   ├── wb_stage                    ← Write Back
│   ├── hazard_unit                 ← Stall, flush, forwarding control
│   └── csr_file                    ← CSR registers + trap logic
├── imem                            ← Instruction memory (BRAM)
└── dmem                            ← Data memory (BRAM)
```

---

## 12. Top-Level Interface

```systemverilog
module rv32imf_top (
    input  logic        clk,          // System clock
    input  logic        rst_n,        // Active-low synchronous reset

    // External interrupt inputs
    input  logic        ext_irq,      // Machine external interrupt
    input  logic        timer_irq,    // Machine timer interrupt
    input  logic        soft_irq,     // Machine software interrupt

    // Debug/status outputs (optional)
    output logic [31:0] debug_pc,     // Current PC value
    output logic        debug_trap    // Asserted when trap taken
);
```

---

## 13. Memory Map

| Base Address | Size | Region |
|-------------|------|--------|
| `0x0000_0000` | 16 KB | Instruction Memory (IMEM) |
| `0x0001_0000` | 16 KB | Data Memory (DMEM) |
| `0x0002_0000` | — | Reserved for MMIO / peripherals |

> The reset vector (PC on `rst_n` de-assertion) is `0x0000_0000`.  
> The stack pointer (`x2`) initial value is `0x0001_FFFC` (top of DMEM, word-aligned).

---

## 14. Verification Plan

### 14.1 Simulation Toolchain
- Simulator: **Vivado Simulator (xsim)** or **Verilator** (for fast simulation)
- Waveform dump: VCD/FSDB for post-sim debug
- Testbench language: SystemVerilog

### 14.2 RISC-V Compliance Tests
- Repository: `riscv-software-src/riscv-tests`
- Test suites to pass:
  - `rv32ui-*` — all RV32I integer tests
  - `rv32um-*` — all M extension tests
  - `rv32uf-*` — all F extension tests
- Each test is loaded into IMEM; processor runs until `tohost` memory-mapped register is written; pass = value 1

### 14.3 Custom Testbenches

| Testbench | Purpose |
|-----------|---------|
| `tb_alu` | Unit test: all ALU operations and edge cases |
| `tb_mul_div` | Unit test: multiply/divide including overflow, div-by-zero |
| `tb_fpu_add` | Unit test: FADD/FSUB with all rounding modes, NaN, Inf, denormals |
| `tb_fpu_mul` | Unit test: FMUL edge cases |
| `tb_fpu_div_sqrt` | Unit test: FDIV, FSQRT including divide-by-zero |
| `tb_fpu_fma` | Unit test: all four FMA variants |
| `tb_fpu_cvt` | Unit test: all integer↔float conversions |
| `tb_hazard` | Integration: RAW hazards, load-use, FPU stalls, branch flushes |
| `tb_exceptions` | Integration: all exception types, MRET, nested traps |
| `tb_csrs` | Unit test: all CSR read/write/set/clear operations |
| `tb_pipeline` | Integration: mixed integer + float program, check register state |
| `tb_top` | Full system test with IMEM + DMEM |

### 14.4 Coverage Goals
- Line/statement coverage: **≥ 95%**
- Branch coverage: **≥ 90%**
- FPU operation coverage: **100%** of opcodes exercised
- Exception coverage: **100%** of mcause codes triggered at least once

---

## 15. File Structure

```
rv32imf/
├── rtl/
│   ├── rv32imf_pkg.sv
│   ├── rv32imf_top.sv
│   ├── rv32imf_core.sv
│   ├── if_stage.sv
│   ├── id_stage.sv
│   │   ├── decoder.sv
│   │   ├── imm_gen.sv
│   │   ├── int_regfile.sv
│   │   └── fp_regfile.sv
│   ├── ex_stage.sv
│   │   ├── alu.sv
│   │   ├── branch_unit.sv
│   │   ├── mul_unit.sv
│   │   ├── div_unit.sv
│   │   └── fpu/
│   │       ├── fpu_top.sv
│   │       ├── fpu_add.sv
│   │       ├── fpu_mul.sv
│   │       ├── fpu_fma.sv
│   │       ├── fpu_div.sv
│   │       ├── fpu_sqrt.sv
│   │       ├── fpu_cvt.sv
│   │       └── fpu_misc.sv
│   ├── mem_stage.sv
│   │   └── load_store_unit.sv
│   ├── wb_stage.sv
│   ├── hazard_unit.sv
│   ├── csr_file.sv
│   ├── imem.sv
│   └── dmem.sv
├── tb/
│   ├── tb_alu.sv
│   ├── tb_mul_div.sv
│   ├── tb_fpu_add.sv
│   ├── tb_fpu_mul.sv
│   ├── tb_fpu_div_sqrt.sv
│   ├── tb_fpu_fma.sv
│   ├── tb_fpu_cvt.sv
│   ├── tb_hazard.sv
│   ├── tb_exceptions.sv
│   ├── tb_csrs.sv
│   ├── tb_pipeline.sv
│   └── tb_top.sv
├── mem/
│   ├── imem_init.mem        ← Hex init file for IMEM
│   └── dmem_init.mem        ← Hex init file for DMEM
├── constraints/
│   └── rv32imf.xdc          ← Vivado timing & pin constraints
├── scripts/
│   ├── run_sim.tcl          ← Vivado sim run script
│   ├── run_compliance.sh    ← riscv-tests runner
│   └── synth.tcl            ← Vivado synthesis script
└── docs/
    └── RV32IMF_Processor_Spec.md
```

---

## Appendix A — Design Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMEM_DEPTH` | 4096 | Instruction memory depth (words) |
| `DMEM_DEPTH` | 4096 | Data memory depth (words) |
| `RESET_VECTOR` | `32'h0000_0000` | PC value after reset |
| `FPU_FLUSH_TO_ZERO` | `1` | Treat denormals as zero (area optimization) |
| `MTVEC_DEFAULT` | `32'h0000_0100` | Default trap vector address |

---

## Appendix B — Key Design Decisions & Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pipeline depth | 5-stage | Classic balance of performance vs. complexity |
| FPU style | Multi-cycle | Area-efficient for FPGA; avoids routing pressure of fully-pipelined FPU |
| Branch resolution | EX stage (2-cycle penalty) | Simple; no branch predictor needed for v1.0 |
| Memory | Harvard BRAM | Natural FPGA fit; no structural hazards between IF and MEM |
| Privilege | M-mode only | Sufficient for bare-metal embedded use; simplifies trap logic |
| Denormals | Flush-to-zero (default) | Saves ~20% FPU area on FPGA; IEEE strict mode available via parameter |
