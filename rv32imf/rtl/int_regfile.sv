// =============================================================================
// int_regfile.sv
// Integer Register File for the RV32IMF processor.
//
// 32 x 32-bit registers (x0 – x31).
// x0 is hardwired to zero — writes to x0 are silently ignored.
//
// Ports:
//   - 2 asynchronous read ports  (rs1, rs2)
//   - 1 synchronous write port   (rd, on rising clock edge)
//
// Asynchronous reads allow the forwarding network to see register values
// without an extra cycle of latency. Reads happen combinationally.
// Writes are clocked so the WB stage commits exactly one cycle.
//
// Read-during-write behaviour:
//   If rs1 or rs2 == rd and we_i is asserted in the same cycle,
//   the OLD value is returned (write-after-read ordering).
//   The hazard unit handles this via forwarding.
// =============================================================================

`timescale 1ns / 1ps

import rv32imf_pkg::*;

module int_regfile (
    input  logic        clk,

    // --- Write port (WB stage) -----------------------------------------------
    input  logic        we_i,       // Write enable
    input  logic [4:0]  rd_i,       // Destination register index
    input  logic [31:0] rd_data_i,  // Data to write

    // --- Read port A (rs1) ---------------------------------------------------
    input  logic [4:0]  rs1_i,      // Source register 1 index
    output logic [31:0] rs1_data_o, // Read data for rs1

    // --- Read port B (rs2) ---------------------------------------------------
    input  logic [4:0]  rs2_i,      // Source register 2 index
    output logic [31:0] rs2_data_o  // Read data for rs2
);

  // ---------------------------------------------------------------------------
  // Register array — index 0 is never written (x0 == 0 always)
  // ---------------------------------------------------------------------------
  logic [31:0] regs [31:1];  // Only x1–x31 are real storage

  // ---------------------------------------------------------------------------
  // Synchronous write (WB stage commits on rising edge)
  // Writes to x0 (rd_i == 0) are suppressed by the if-condition.
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (we_i && rd_i != 5'b0)
      regs[rd_i] <= rd_data_i;
  end

  // ---------------------------------------------------------------------------
  // Asynchronous reads — x0 always returns 0
  // ---------------------------------------------------------------------------
  assign rs1_data_o = (rs1_i == 5'b0) ? 32'b0 : regs[rs1_i];
  assign rs2_data_o = (rs2_i == 5'b0) ? 32'b0 : regs[rs2_i];

endmodule