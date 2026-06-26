// =============================================================================
// alu.sv
// Integer Arithmetic Logic Unit for the RV32IMF processor.
//
// Purely combinational. Supports all RV32I ALU operations.
// Used in the EX stage for:
//   - Register-register instructions (OP_REG)
//   - Register-immediate instructions (OP_IMM)
//   - Address calculation (loads, stores, AUIPC)
//   - LUI (pass-through of immediate via ALU_PASS)
//   - JAL/JALR target calculation (done in branch_unit; ALU computes PC+4)
//
// NOTE: MUL/DIV are handled by separate mul_unit / div_unit modules,
//       not by this ALU.
// =============================================================================

`timescale 1ns / 1ps

import rv32imf_pkg::*;

module alu (
    // --- Operands ------------------------------------------------------------
    input  logic [31:0] src_a,    // Operand A (register value or PC)
    input  logic [31:0] src_b,    // Operand B (register value or immediate)

    // --- Operation select ----------------------------------------------------
    input  alu_op_t     alu_op,   // Operation (see rv32imf_pkg::alu_op_t)

    // --- Outputs -------------------------------------------------------------
    output logic [31:0] result,   // Computed result
    output logic        zero      // 1 when result == 0 (used by branch_unit)
);

  // ---------------------------------------------------------------------------
  // Main ALU logic — purely combinational
  // ---------------------------------------------------------------------------
  always_comb begin
    unique case (alu_op)
      // Arithmetic
      ALU_ADD  : result = src_a + src_b;
      ALU_SUB  : result = src_a - src_b;

      // Logical
      ALU_AND  : result = src_a & src_b;
      ALU_OR   : result = src_a | src_b;
      ALU_XOR  : result = src_a ^ src_b;

      // Shifts — only low 5 bits of src_b are used as shift amount (RV32 spec)
      ALU_SLL  : result = src_a << src_b[4:0];
      ALU_SRL  : result = src_a >> src_b[4:0];
      ALU_SRA  : result = $signed(src_a) >>> src_b[4:0];  // arithmetic (sign-extend)

      // Set-Less-Than
      ALU_SLT  : result = {{31{1'b0}}, ($signed(src_a) < $signed(src_b))};
      ALU_SLTU : result = {{31{1'b0}}, (src_a < src_b)};

      // Pass-through src_b (used for LUI: rd = imm, so src_a=0, src_b=imm)
      ALU_PASS : result = src_b;

      // Safety default (should never be reached with valid alu_op)
      default  : result = 32'b0;
    endcase
  end

  // ---------------------------------------------------------------------------
  // Zero flag — asserted when result is all-zero
  // Used by branch_unit to evaluate BEQ / BNE conditions.
  // ---------------------------------------------------------------------------
  assign zero = (result == 32'b0);

endmodule