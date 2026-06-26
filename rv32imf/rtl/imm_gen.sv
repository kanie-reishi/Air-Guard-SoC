// =============================================================================
// imm_gen.sv
// Immediate Generator for the RV32IMF processor.
//
// Extracts and sign-extends the immediate value from the raw instruction word
// based on the RV32 immediate encoding format.
//
// Formats supported:
//   I-type : ADDI, LOAD, JALR, CSR (sign-extended 12-bit)
//   S-type : STORE                  (sign-extended 12-bit, split field)
//   B-type : BRANCH                 (sign-extended 13-bit, bit 0 always 0)
//   U-type : LUI, AUIPC             (32-bit, lower 12 bits zeroed)
//   J-type : JAL                    (sign-extended 21-bit, bit 0 always 0)
//   Z-type : CSR zimm               (zero-extended 5-bit from rs1 field)
//
// Purely combinational — no clock required.
// =============================================================================

`timescale 1ns / 1ps

import rv32imf_pkg::*;

module imm_gen (
    input  logic [31:0] instr_i,   // Raw 32-bit instruction word
    input  imm_type_t   imm_type_i, // Immediate format select (from decoder)
    output logic [31:0] imm_o      // Sign/zero-extended 32-bit immediate
);

  always_comb begin
    unique case (imm_type_i)

      // -----------------------------------------------------------------------
      // I-type: instr[31:20] → sign-extended 12-bit
      // Used by: ADDI, SLTI, XORI, ORI, ANDI, SLLI, SRLI, SRAI,
      //          LB, LH, LW, LBU, LHU, JALR, FLW
      // -----------------------------------------------------------------------
      IMM_I: imm_o = {{20{instr_i[31]}}, instr_i[31:20]};

      // -----------------------------------------------------------------------
      // S-type: {instr[31:25], instr[11:7]} → sign-extended 12-bit
      // Used by: SB, SH, SW, FSW
      // -----------------------------------------------------------------------
      IMM_S: imm_o = {{20{instr_i[31]}}, instr_i[31:25], instr_i[11:7]};

      // -----------------------------------------------------------------------
      // B-type: {instr[31], instr[7], instr[30:25], instr[11:8], 1'b0}
      //         → sign-extended 13-bit (LSB always 0, half-word aligned)
      // Used by: BEQ, BNE, BLT, BGE, BLTU, BGEU
      // -----------------------------------------------------------------------
      IMM_B: imm_o = {{19{instr_i[31]}},
                       instr_i[31],
                       instr_i[7],
                       instr_i[30:25],
                       instr_i[11:8],
                       1'b0};

      // -----------------------------------------------------------------------
      // U-type: instr[31:12] placed in upper 20 bits, lower 12 bits zeroed
      // Used by: LUI, AUIPC
      // -----------------------------------------------------------------------
      IMM_U: imm_o = {instr_i[31:12], 12'b0};

      // -----------------------------------------------------------------------
      // J-type: {instr[31], instr[19:12], instr[20], instr[30:21], 1'b0}
      //         → sign-extended 21-bit (LSB always 0, half-word aligned)
      // Used by: JAL
      // -----------------------------------------------------------------------
      IMM_J: imm_o = {{11{instr_i[31]}},
                       instr_i[31],
                       instr_i[19:12],
                       instr_i[20],
                       instr_i[30:21],
                       1'b0};

      // -----------------------------------------------------------------------
      // Z-type: zero-extended 5-bit from rs1 field (instr[19:15])
      // Used by: CSRRWI, CSRRSI, CSRRCI (CSR immediate variants)
      // -----------------------------------------------------------------------
      IMM_Z: imm_o = {27'b0, instr_i[19:15]};

      // Safety default
      default: imm_o = 32'b0;

    endcase
  end

endmodule