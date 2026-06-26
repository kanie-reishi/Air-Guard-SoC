// =============================================================================
// fp_regfile.sv
// Floating-Point Register File for the RV32IMF processor.
//
// 32 x 32-bit registers (f0 – f31).
// Unlike the integer register file, there is NO hardwired-zero register —
// all 32 entries are general-purpose writable storage.
//
// Ports:
//   - 3 asynchronous read ports  (rs1, rs2, rs3)
//     rs3 is required for fused multiply-add instructions:
//     FMADD.S  rd = rs1*rs2 + rs3
//     FMSUB.S  rd = rs1*rs2 - rs3
//     FNMADD.S rd = -(rs1*rs2 + rs3)
//     FNMSUB.S rd = -(rs1*rs2 - rs3)
//   - 1 synchronous write port   (rd, on rising clock edge)
//
// Read-during-write: returns OLD value (same policy as int_regfile).
// The hazard unit manages FPU result forwarding.
// =============================================================================

`timescale 1ns / 1ps

import rv32imf_pkg::*;

module fp_regfile (
    input  logic        clk,

    // --- Write port (WB stage) -----------------------------------------------
    input  logic        we_i,       // Write enable
    input  logic [4:0]  rd_i,       // Destination register index (f0–f31)
    input  logic [31:0] rd_data_i,  // Data to write (IEEE 754 single)

    // --- Read port A (rs1) ---------------------------------------------------
    input  logic [4:0]  rs1_i,      // Source register 1 index
    output logic [31:0] rs1_data_o, // Read data for rs1

    // --- Read port B (rs2) ---------------------------------------------------
    input  logic [4:0]  rs2_i,      // Source register 2 index
    output logic [31:0] rs2_data_o, // Read data for rs2

    // --- Read port C (rs3) — FMA only ----------------------------------------
    input  logic [4:0]  rs3_i,      // Source register 3 index
    output logic [31:0] rs3_data_o  // Read data for rs3
);

  // ---------------------------------------------------------------------------
  // Register array — all 32 entries are real storage (no hardwired zero)
  // ---------------------------------------------------------------------------
  logic [31:0] regs [31:0];

  // ---------------------------------------------------------------------------
  // Synchronous write (WB stage commits on rising edge)
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (we_i)
      regs[rd_i] <= rd_data_i;
  end

  // ---------------------------------------------------------------------------
  // Asynchronous reads — all three ports are independent
  // ---------------------------------------------------------------------------
  assign rs1_data_o = regs[rs1_i];
  assign rs2_data_o = regs[rs2_i];
  assign rs3_data_o = regs[rs3_i];

endmodule