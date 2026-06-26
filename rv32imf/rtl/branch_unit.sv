// =============================================================================
// branch_unit.sv
// Branch & Jump resolution unit for the RV32IMF processor.
//
// Sits in the EX stage. Purely combinational.
// Responsibilities:
//   1. Evaluate branch conditions (BEQ/BNE/BLT/BGE/BLTU/BGEU)
//   2. Compute the redirect target PC for taken branches and jumps
//   3. Assert branch_taken_o whenever the PC must be redirected
//
// Target address computation:
//   BRANCH  : PC + imm          (PC-relative, imm from B-type encoding)
//   JAL     : PC + imm          (already computed by ALU; passed through)
//   JALR    : (rs1 + imm) & ~1  (ALU computes rs1+imm; we clear LSB here)
//
// When branch_taken_o is asserted, the hazard_unit flushes the IF and ID
// pipeline registers (2-cycle branch penalty, static not-taken prediction).
// =============================================================================

`timescale 1ns / 1ps

import rv32imf_pkg::*;

module branch_unit (
    // --- Operands (post-forwarding register values) --------------------------
    input  logic [31:0] rs1_i,          // Forwarded rs1 value
    input  logic [31:0] rs2_i,          // Forwarded rs2 value

    // --- Instruction context -------------------------------------------------
    input  logic [2:0]  funct3_i,       // Branch type (F3_BEQ etc.)
    input  logic        is_branch_i,    // Instruction is a conditional branch
    input  logic        is_jal_i,       // Instruction is JAL
    input  logic        is_jalr_i,      // Instruction is JALR

    // --- PC and immediate ----------------------------------------------------
    input  logic [31:0] pc_i,           // PC of the branch/jump instruction
    input  logic [31:0] imm_i,          // Sign-extended immediate from imm_gen

    // --- ALU result (used for JALR target = rs1 + imm) ----------------------
    input  logic [31:0] alu_result_i,   // ALU output (rs1 + imm for JALR)

    // --- Outputs -------------------------------------------------------------
    output logic        branch_taken_o, // 1 = redirect PC; 0 = fall through
    output logic [31:0] pc_target_o     // Redirect target address
);

  // ---------------------------------------------------------------------------
  // Branch condition evaluation
  // Direct comparison on forwarded operands — avoids relying on the ALU zero
  // flag and correctly handles all signed/unsigned cases in one place.
  // ---------------------------------------------------------------------------
  logic branch_cond;

  always_comb begin
    unique case (funct3_i)
      F3_BEQ  : branch_cond = (rs1_i == rs2_i);
      F3_BNE  : branch_cond = (rs1_i != rs2_i);
      F3_BLT  : branch_cond = ($signed(rs1_i) <  $signed(rs2_i));
      F3_BGE  : branch_cond = ($signed(rs1_i) >= $signed(rs2_i));
      F3_BLTU : branch_cond = (rs1_i <  rs2_i);
      F3_BGEU : branch_cond = (rs1_i >= rs2_i);
      default : branch_cond = 1'b0;
    endcase
  end

  // ---------------------------------------------------------------------------
  // branch_taken: asserted for any instruction that redirects the PC
  //   - Conditional branch whose condition evaluates true
  //   - JAL  (unconditional, always taken)
  //   - JALR (unconditional, always taken)
  // ---------------------------------------------------------------------------
  assign branch_taken_o = (is_branch_i & branch_cond) | is_jal_i | is_jalr_i;

  // ---------------------------------------------------------------------------
  // PC target computation
  //
  //   BRANCH / JAL : PC + imm
  //     The decoder sets alu_src_a_pc for JAL so the ALU already computed
  //     PC+imm, but we recompute independently here for clarity and to keep
  //     the branch unit self-contained.
  //
  //   JALR         : (rs1 + imm) & ~32'h1
  //     LSB is forced to 0 per the RV32 spec to ensure instruction alignment.
  //     rs1 + imm has already been computed by the ALU (alu_result_i).
  // ---------------------------------------------------------------------------
  always_comb begin
    if (is_jalr_i)
      pc_target_o = {alu_result_i[31:1], 1'b0}; // clear LSB
    else
      pc_target_o = pc_i + imm_i;               // branch or JAL: PC-relative
  end

endmodule