// =============================================================================
// decoder.sv
// Instruction Decoder for the RV32IMF processor.
//
// Purely combinational. Takes the raw 32-bit instruction and produces all
// control signals consumed by the pipeline stages.
//
// Outputs are registered in the ID/EX pipeline register (in id_stage).
// Any unrecognised instruction asserts illegal_instr_o, which triggers
// an Illegal Instruction exception in the trap/CSR logic.
// =============================================================================

`timescale 1ns / 1ps

import rv32imf_pkg::*;

module decoder (
    // --- Instruction word ----------------------------------------------------
    input  logic [31:0] instr_i,        // Raw instruction from IF/ID register

    // --- Decoded fields (passed to other ID sub-modules) ---------------------
    output logic [4:0]  rs1_o,          // Source register 1 index
    output logic [4:0]  rs2_o,          // Source register 2 index
    output logic [4:0]  rs3_o,          // Source register 3 index (FMA only)
    output logic [4:0]  rd_o,           // Destination register index
    output logic [2:0]  funct3_o,       // funct3 field (passed to mem/branch)
    output imm_type_t   imm_type_o,     // Immediate format for imm_gen

    // --- ALU control ---------------------------------------------------------
    output alu_op_t     alu_op_o,       // ALU operation
    output logic        alu_src_a_pc_o, // 1 = src_a is PC (AUIPC, JAL)
    output logic        alu_src_b_imm_o,// 1 = src_b is immediate (I/S/U/J-type)

    // --- Register file write control -----------------------------------------
    output logic        int_reg_we_o,   // Write enable for integer register file
    output logic        fp_reg_we_o,    // Write enable for float register file
    output wb_sel_t     wb_sel_o,       // Write-back data source select

    // --- Memory control ------------------------------------------------------
    output logic        mem_re_o,       // Data memory read  enable (loads)
    output logic        mem_we_o,       // Data memory write enable (stores)
    output logic        mem_fp_o,       // 1 = memory op is float (FLW/FSW)

    // --- Branch / Jump control -----------------------------------------------
    output logic        is_branch_o,    // Instruction is a conditional branch
    output logic        is_jal_o,       // Instruction is JAL
    output logic        is_jalr_o,      // Instruction is JALR

    // --- FPU control ---------------------------------------------------------
    output logic        fpu_valid_o,    // Instruction dispatched to FPU
    output fpu_op_t     fpu_op_o,       // FPU operation code
    output logic [2:0]  fpu_rm_o,       // Rounding mode (funct3 or dynamic)

    // --- System / CSR control ------------------------------------------------
    output logic        is_csr_o,       // Instruction is a CSR access
    output logic        is_ecall_o,     // ECALL
    output logic        is_ebreak_o,    // EBREAK
    output logic        is_mret_o,      // MRET

    // --- Exception -----------------------------------------------------------
    output logic        illegal_instr_o // Unrecognised / unsupported instruction
);

  // ---------------------------------------------------------------------------
  // Instruction field extraction
  // ---------------------------------------------------------------------------
  logic [6:0] opcode;
  logic [2:0] funct3;
  logic [6:0] funct7;
  logic [4:0] rs1, rs2, rs3, rd;

  assign opcode = instr_i[6:0];
  assign rd     = instr_i[11:7];
  assign funct3 = instr_i[14:12];
  assign rs1    = instr_i[19:15];
  assign rs2    = instr_i[24:20];
  assign rs3    = instr_i[31:27];
  assign funct7 = instr_i[31:25];

  // Pass decoded fields out
  assign rs1_o    = rs1;
  assign rs2_o    = rs2;
  assign rs3_o    = rs3;
  assign rd_o     = rd;
  assign funct3_o = funct3;

  // ---------------------------------------------------------------------------
  // Default signal values (overridden per instruction below)
  // ---------------------------------------------------------------------------
  always_comb begin
    // Defaults
    imm_type_o      = IMM_I;
    alu_op_o        = ALU_ADD;
    alu_src_a_pc_o  = 1'b0;
    alu_src_b_imm_o = 1'b0;
    int_reg_we_o    = 1'b0;
    fp_reg_we_o     = 1'b0;
    wb_sel_o        = WB_ALU;
    mem_re_o        = 1'b0;
    mem_we_o        = 1'b0;
    mem_fp_o        = 1'b0;
    is_branch_o     = 1'b0;
    is_jal_o        = 1'b0;
    is_jalr_o       = 1'b0;
    fpu_valid_o     = 1'b0;
    fpu_op_o        = FPU_ADD;
    fpu_rm_o        = funct3;   // default: use instruction's rm field
    is_csr_o        = 1'b0;
    is_ecall_o      = 1'b0;
    is_ebreak_o     = 1'b0;
    is_mret_o       = 1'b0;
    illegal_instr_o = 1'b0;

    unique case (opcode)

      // -----------------------------------------------------------------------
      // LUI — Load Upper Immediate
      // rd = imm[31:12] << 12  (ALU_PASS passes src_b=imm through)
      // -----------------------------------------------------------------------
      OP_LUI: begin
        imm_type_o      = IMM_U;
        alu_op_o        = ALU_PASS;
        alu_src_b_imm_o = 1'b1;
        int_reg_we_o    = 1'b1;
        wb_sel_o        = WB_ALU;
      end

      // -----------------------------------------------------------------------
      // AUIPC — Add Upper Immediate to PC
      // rd = PC + imm[31:12] << 12
      // -----------------------------------------------------------------------
      OP_AUIPC: begin
        imm_type_o      = IMM_U;
        alu_op_o        = ALU_ADD;
        alu_src_a_pc_o  = 1'b1;   // src_a = PC
        alu_src_b_imm_o = 1'b1;   // src_b = imm
        int_reg_we_o    = 1'b1;
        wb_sel_o        = WB_ALU;
      end

      // -----------------------------------------------------------------------
      // JAL — Jump and Link
      // rd = PC+4;  PC = PC + imm
      // -----------------------------------------------------------------------
      OP_JAL: begin
        imm_type_o      = IMM_J;
        alu_op_o        = ALU_ADD;
        alu_src_a_pc_o  = 1'b1;   // target = PC + imm
        alu_src_b_imm_o = 1'b1;
        int_reg_we_o    = 1'b1;
        wb_sel_o        = WB_PC4; // rd = PC+4
        is_jal_o        = 1'b1;
      end

      // -----------------------------------------------------------------------
      // JALR — Jump and Link Register
      // rd = PC+4;  PC = (rs1 + imm) & ~1
      // -----------------------------------------------------------------------
      OP_JALR: begin
        imm_type_o      = IMM_I;
        alu_op_o        = ALU_ADD; // target = rs1 + imm (LSB cleared in EX)
        alu_src_b_imm_o = 1'b1;
        int_reg_we_o    = 1'b1;
        wb_sel_o        = WB_PC4; // rd = PC+4
        is_jalr_o       = 1'b1;
      end

      // -----------------------------------------------------------------------
      // BRANCH — BEQ, BNE, BLT, BGE, BLTU, BGEU
      // No register write. branch_unit evaluates condition in EX.
      // -----------------------------------------------------------------------
      OP_BRANCH: begin
        imm_type_o      = IMM_B;
        alu_op_o        = ALU_SUB; // branch_unit uses ALU zero/sign for compare
        alu_src_a_pc_o  = 1'b0;
        alu_src_b_imm_o = 1'b0;
        is_branch_o     = 1'b1;
        // Validate funct3
        if (funct3 != F3_BEQ  && funct3 != F3_BNE  &&
            funct3 != F3_BLT  && funct3 != F3_BGE  &&
            funct3 != F3_BLTU && funct3 != F3_BGEU)
          illegal_instr_o = 1'b1;
      end

      // -----------------------------------------------------------------------
      // LOAD — LB, LH, LW, LBU, LHU
      // rd = Mem[rs1 + imm]
      // -----------------------------------------------------------------------
      OP_LOAD: begin
        imm_type_o      = IMM_I;
        alu_op_o        = ALU_ADD; // address = rs1 + imm
        alu_src_b_imm_o = 1'b1;
        int_reg_we_o    = 1'b1;
        wb_sel_o        = WB_MEM;
        mem_re_o        = 1'b1;
        // Validate funct3
        if (funct3 != F3_LB  && funct3 != F3_LH  && funct3 != F3_LW &&
            funct3 != F3_LBU && funct3 != F3_LHU)
          illegal_instr_o = 1'b1;
      end

      // -----------------------------------------------------------------------
      // STORE — SB, SH, SW
      // Mem[rs1 + imm] = rs2
      // -----------------------------------------------------------------------
      OP_STORE: begin
        imm_type_o      = IMM_S;
        alu_op_o        = ALU_ADD; // address = rs1 + imm
        alu_src_b_imm_o = 1'b1;
        mem_we_o        = 1'b1;
        // Validate funct3
        if (funct3 != F3_SB && funct3 != F3_SH && funct3 != F3_SW)
          illegal_instr_o = 1'b1;
      end

      // -----------------------------------------------------------------------
      // OP_IMM — Register-Immediate arithmetic
      // ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
      // -----------------------------------------------------------------------
      OP_IMM: begin
        imm_type_o      = IMM_I;
        alu_src_b_imm_o = 1'b1;
        int_reg_we_o    = 1'b1;
        wb_sel_o        = WB_ALU;
        unique case (funct3)
          F3_ADD_SUB: alu_op_o = ALU_ADD;           // ADDI
          F3_SLT    : alu_op_o = ALU_SLT;           // SLTI
          F3_SLTU   : alu_op_o = ALU_SLTU;          // SLTIU
          F3_XOR    : alu_op_o = ALU_XOR;           // XORI
          F3_OR     : alu_op_o = ALU_OR;            // ORI
          F3_AND    : alu_op_o = ALU_AND;           // ANDI
          F3_SLL: begin
            alu_op_o = ALU_SLL;                     // SLLI
            if (funct7 != F7_NORMAL) illegal_instr_o = 1'b1;
          end
          F3_SRL_SRA: begin
            if      (funct7 == F7_NORMAL) alu_op_o = ALU_SRL; // SRLI
            else if (funct7 == F7_ALT)    alu_op_o = ALU_SRA; // SRAI
            else                          illegal_instr_o = 1'b1;
          end
          default: illegal_instr_o = 1'b1;
        endcase
      end

      // -----------------------------------------------------------------------
      // OP_REG — Register-Register (RV32I + M extension)
      // ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND
      // MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
      // -----------------------------------------------------------------------
      OP_REG: begin
        int_reg_we_o    = 1'b1;
        wb_sel_o        = WB_ALU;
        alu_src_b_imm_o = 1'b0;
        if (funct7 == F7_MEXT) begin
          // M extension — mul_unit / div_unit handle execution
          // alu_op_o is unused for M-ext; EX stage checks funct3 directly
          alu_op_o = ALU_ADD; // placeholder
        end else begin
          unique case (funct3)
            F3_ADD_SUB: begin
              if      (funct7 == F7_NORMAL) alu_op_o = ALU_ADD;
              else if (funct7 == F7_ALT)    alu_op_o = ALU_SUB;
              else                          illegal_instr_o = 1'b1;
            end
            F3_SLL: begin
              alu_op_o = ALU_SLL;
              if (funct7 != F7_NORMAL) illegal_instr_o = 1'b1;
            end
            F3_SLT: begin
              alu_op_o = ALU_SLT;
              if (funct7 != F7_NORMAL) illegal_instr_o = 1'b1;
            end
            F3_SLTU: begin
              alu_op_o = ALU_SLTU;
              if (funct7 != F7_NORMAL) illegal_instr_o = 1'b1;
            end
            F3_XOR: begin
              alu_op_o = ALU_XOR;
              if (funct7 != F7_NORMAL) illegal_instr_o = 1'b1;
            end
            F3_SRL_SRA: begin
              if      (funct7 == F7_NORMAL) alu_op_o = ALU_SRL;
              else if (funct7 == F7_ALT)    alu_op_o = ALU_SRA;
              else                          illegal_instr_o = 1'b1;
            end
            F3_OR: begin
              alu_op_o = ALU_OR;
              if (funct7 != F7_NORMAL) illegal_instr_o = 1'b1;
            end
            F3_AND: begin
              alu_op_o = ALU_AND;
              if (funct7 != F7_NORMAL) illegal_instr_o = 1'b1;
            end
            default: illegal_instr_o = 1'b1;
          endcase
        end
      end

      // -----------------------------------------------------------------------
      // SYSTEM — ECALL, EBREAK, MRET, CSR*
      // -----------------------------------------------------------------------
      OP_SYSTEM: begin
        unique case (funct3)
          F3_ECALL: begin
            // Disambiguate by instr[31:20]
            unique case (instr_i[31:20])
              12'b0000_0000_0000: is_ecall_o  = 1'b1; // ECALL
              12'b0000_0000_0001: is_ebreak_o = 1'b1; // EBREAK
              12'b0011_0000_0010: is_mret_o   = 1'b1; // MRET
              default:            illegal_instr_o = 1'b1;
            endcase
          end
          // CSR instructions — all write to integer rd
          F3_CSRRW, F3_CSRRS, F3_CSRRC: begin
            is_csr_o     = 1'b1;
            int_reg_we_o = 1'b1;
            wb_sel_o     = WB_ALU; // CSR result routed via ALU path
            imm_type_o   = IMM_I;  // rs1 used as register operand
          end
          F3_CSRRWI, F3_CSRRSI, F3_CSRRCI: begin
            is_csr_o        = 1'b1;
            int_reg_we_o    = 1'b1;
            wb_sel_o        = WB_ALU;
            imm_type_o      = IMM_Z;  // zimm from instr[19:15]
            alu_src_b_imm_o = 1'b1;
          end
          default: illegal_instr_o = 1'b1;
        endcase
      end

      // -----------------------------------------------------------------------
      // FLW — Float Load Word
      // fd = Mem[rs1 + imm]   (32-bit float load)
      // -----------------------------------------------------------------------
      OP_FLOAD: begin
        imm_type_o      = IMM_I;
        alu_op_o        = ALU_ADD; // address = rs1 + imm
        alu_src_b_imm_o = 1'b1;
        fp_reg_we_o     = 1'b1;
        wb_sel_o        = WB_MEM;
        mem_re_o        = 1'b1;
        mem_fp_o        = 1'b1;
        if (funct3 != 3'b010) illegal_instr_o = 1'b1; // only .S (word) supported
      end

      // -----------------------------------------------------------------------
      // FSW — Float Store Word
      // Mem[rs1 + imm] = fs2
      // -----------------------------------------------------------------------
      OP_FSTORE: begin
        imm_type_o      = IMM_S;
        alu_op_o        = ALU_ADD; // address = rs1 + imm
        alu_src_b_imm_o = 1'b1;
        mem_we_o        = 1'b1;
        mem_fp_o        = 1'b1;
        if (funct3 != 3'b010) illegal_instr_o = 1'b1;
      end

      // -----------------------------------------------------------------------
      // FMADD.S — fd = rs1*rs2 + rs3
      // -----------------------------------------------------------------------
      OP_FMADD: begin
        fpu_valid_o  = 1'b1;
        fpu_op_o     = FPU_FMADD;
        fpu_rm_o     = funct3;
        fp_reg_we_o  = 1'b1;
        wb_sel_o     = WB_FPU;
        if (instr_i[26:25] != 2'b00) illegal_instr_o = 1'b1; // fmt must be S
      end

      // -----------------------------------------------------------------------
      // FMSUB.S — fd = rs1*rs2 - rs3
      // -----------------------------------------------------------------------
      OP_FMSUB: begin
        fpu_valid_o  = 1'b1;
        fpu_op_o     = FPU_FMSUB;
        fpu_rm_o     = funct3;
        fp_reg_we_o  = 1'b1;
        wb_sel_o     = WB_FPU;
        if (instr_i[26:25] != 2'b00) illegal_instr_o = 1'b1;
      end

      // -----------------------------------------------------------------------
      // FNMSUB.S — fd = -(rs1*rs2 - rs3)
      // -----------------------------------------------------------------------
      OP_FNMSUB: begin
        fpu_valid_o  = 1'b1;
        fpu_op_o     = FPU_FNMSUB;
        fpu_rm_o     = funct3;
        fp_reg_we_o  = 1'b1;
        wb_sel_o     = WB_FPU;
        if (instr_i[26:25] != 2'b00) illegal_instr_o = 1'b1;
      end

      // -----------------------------------------------------------------------
      // FNMADD.S — fd = -(rs1*rs2 + rs3)
      // -----------------------------------------------------------------------
      OP_FNMADD: begin
        fpu_valid_o  = 1'b1;
        fpu_op_o     = FPU_FNMADD;
        fpu_rm_o     = funct3;
        fp_reg_we_o  = 1'b1;
        wb_sel_o     = WB_FPU;
        if (instr_i[26:25] != 2'b00) illegal_instr_o = 1'b1;
      end

      // -----------------------------------------------------------------------
      // OP_FP — All remaining F-extension instructions
      // Disambiguated by funct7 (instr[31:25])
      // -----------------------------------------------------------------------
      OP_FP: begin
        fpu_rm_o = funct3;
        unique case (funct7)
          7'b000_0000: begin // FADD.S
            fpu_valid_o = 1'b1; fpu_op_o = FPU_ADD;
            fp_reg_we_o = 1'b1; wb_sel_o = WB_FPU;
          end
          7'b000_0100: begin // FSUB.S
            fpu_valid_o = 1'b1; fpu_op_o = FPU_SUB;
            fp_reg_we_o = 1'b1; wb_sel_o = WB_FPU;
          end
          7'b000_1000: begin // FMUL.S
            fpu_valid_o = 1'b1; fpu_op_o = FPU_MUL;
            fp_reg_we_o = 1'b1; wb_sel_o = WB_FPU;
          end
          7'b000_1100: begin // FDIV.S
            fpu_valid_o = 1'b1; fpu_op_o = FPU_DIV;
            fp_reg_we_o = 1'b1; wb_sel_o = WB_FPU;
          end
          7'b010_1100: begin // FSQRT.S (rs2 must be 5'b00000)
            fpu_valid_o = 1'b1; fpu_op_o = FPU_SQRT;
            fp_reg_we_o = 1'b1; wb_sel_o = WB_FPU;
            if (rs2 != 5'b00000) illegal_instr_o = 1'b1;
          end
          7'b001_0000: begin // FSGNJ.S / FSGNJN.S / FSGNJX.S
            fp_reg_we_o = 1'b1; wb_sel_o = WB_FPU; fpu_valid_o = 1'b1;
            unique case (funct3)
              3'b000: fpu_op_o = FPU_SGNJ;
              3'b001: fpu_op_o = FPU_SGNJN;
              3'b010: fpu_op_o = FPU_SGNJX;
              default: illegal_instr_o = 1'b1;
            endcase
          end
          7'b001_0100: begin // FMIN.S / FMAX.S
            fp_reg_we_o = 1'b1; wb_sel_o = WB_FPU; fpu_valid_o = 1'b1;
            unique case (funct3)
              3'b000: fpu_op_o = FPU_MIN;
              3'b001: fpu_op_o = FPU_MAX;
              default: illegal_instr_o = 1'b1;
            endcase
          end
          7'b110_0000: begin // FCVT.W.S / FCVT.WU.S → integer rd
            fpu_valid_o  = 1'b1;
            int_reg_we_o = 1'b1; wb_sel_o = WB_FPU;
            unique case (rs2)
              5'b00000: fpu_op_o = FPU_CVT_WS;   // FCVT.W.S
              5'b00001: fpu_op_o = FPU_CVT_WUS;  // FCVT.WU.S
              default:  illegal_instr_o = 1'b1;
            endcase
          end
          7'b110_1000: begin // FCVT.S.W / FCVT.S.WU → float rd
            fpu_valid_o = 1'b1;
            fp_reg_we_o = 1'b1; wb_sel_o = WB_FPU;
            unique case (rs2)
              5'b00000: fpu_op_o = FPU_CVT_SW;   // FCVT.S.W
              5'b00001: fpu_op_o = FPU_CVT_SWU;  // FCVT.S.WU
              default:  illegal_instr_o = 1'b1;
            endcase
          end
          7'b111_0000: begin // FMV.X.W / FCLASS.S → integer rd
            int_reg_we_o = 1'b1; wb_sel_o = WB_FPU; fpu_valid_o = 1'b1;
            unique case (funct3)
              3'b000: fpu_op_o = FPU_MVX2W;  // FMV.X.W
              3'b001: fpu_op_o = FPU_CLASS;  // FCLASS.S
              default: illegal_instr_o = 1'b1;
            endcase
            if (rs2 != 5'b00000) illegal_instr_o = 1'b1;
          end
          7'b111_1000: begin // FMV.W.X → float rd
            fpu_valid_o = 1'b1; fpu_op_o = FPU_MVW2X;
            fp_reg_we_o = 1'b1; wb_sel_o = WB_FPU;
            if (rs2 != 5'b00000 || funct3 != 3'b000)
              illegal_instr_o = 1'b1;
          end
          7'b101_0000: begin // FEQ.S / FLT.S / FLE.S → integer rd
            fpu_valid_o  = 1'b1;
            int_reg_we_o = 1'b1; wb_sel_o = WB_FPU;
            unique case (funct3)
              3'b010: fpu_op_o = FPU_EQ;
              3'b001: fpu_op_o = FPU_LT;
              3'b000: fpu_op_o = FPU_LE;
              default: illegal_instr_o = 1'b1;
            endcase
          end
          default: illegal_instr_o = 1'b1;
        endcase
      end

      // -----------------------------------------------------------------------
      // Any unrecognised opcode
      // -----------------------------------------------------------------------
      default: illegal_instr_o = 1'b1;

    endcase
  end

endmodule