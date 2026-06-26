// =============================================================================
// rv32imf_pkg.sv
// Shared types, enums, and constants for the RV32IMF processor.
// Imported by all modules in the design.
// =============================================================================

package rv32imf_pkg;

  // ---------------------------------------------------------------------------
  // Instruction Opcodes (bits [6:0])
  // ---------------------------------------------------------------------------
  typedef enum logic [6:0] {
    OP_LUI    = 7'b011_0111,  // Load Upper Immediate
    OP_AUIPC  = 7'b001_0111,  // Add Upper Immediate to PC
    OP_JAL    = 7'b110_1111,  // Jump and Link
    OP_JALR   = 7'b110_0111,  // Jump and Link Register
    OP_BRANCH = 7'b110_0011,  // Branch (BEQ/BNE/BLT/BGE/BLTU/BGEU)
    OP_LOAD   = 7'b000_0011,  // Integer Load (LB/LH/LW/LBU/LHU)
    OP_STORE  = 7'b010_0011,  // Integer Store (SB/SH/SW)
    OP_IMM    = 7'b001_0011,  // Integer Register-Immediate (ADDI etc.)
    OP_REG    = 7'b011_0011,  // Integer Register-Register (ADD/SUB/MUL etc.)
    OP_SYSTEM = 7'b111_0011,  // System (ECALL/EBREAK/MRET/CSR*)
    OP_FLOAD  = 7'b000_0111,  // Float Load  (FLW)
    OP_FSTORE = 7'b010_0111,  // Float Store (FSW)
    OP_FMADD  = 7'b100_0011,  // FMADD.S
    OP_FMSUB  = 7'b100_0111,  // FMSUB.S
    OP_FNMSUB = 7'b100_1011,  // FNMSUB.S
    OP_FNMADD = 7'b100_1111,  // FNMADD.S
    OP_FP     = 7'b101_0011   // All other FP ops (FADD/FSUB/FMUL/FDIV/etc.)
  } opcode_t;

  // ---------------------------------------------------------------------------
  // ALU Operation Codes
  // ---------------------------------------------------------------------------
  typedef enum logic [3:0] {
    ALU_ADD  = 4'h0,  // Addition              : src_a + src_b
    ALU_SUB  = 4'h1,  // Subtraction           : src_a - src_b
    ALU_AND  = 4'h2,  // Bitwise AND           : src_a & src_b
    ALU_OR   = 4'h3,  // Bitwise OR            : src_a | src_b
    ALU_XOR  = 4'h4,  // Bitwise XOR           : src_a ^ src_b
    ALU_SLL  = 4'h5,  // Shift Left Logical    : src_a << src_b[4:0]
    ALU_SRL  = 4'h6,  // Shift Right Logical   : src_a >> src_b[4:0]
    ALU_SRA  = 4'h7,  // Shift Right Arithmetic: src_a >>> src_b[4:0]
    ALU_SLT  = 4'h8,  // Set Less Than (signed)
    ALU_SLTU = 4'h9,  // Set Less Than Unsigned
    ALU_PASS = 4'hA   // Pass src_b through (used for LUI)
  } alu_op_t;

  // ---------------------------------------------------------------------------
  // FPU Operation Codes
  // ---------------------------------------------------------------------------
  typedef enum logic [4:0] {
    FPU_ADD    = 5'd0,   // FADD.S
    FPU_SUB    = 5'd1,   // FSUB.S
    FPU_MUL    = 5'd2,   // FMUL.S
    FPU_DIV    = 5'd3,   // FDIV.S
    FPU_SQRT   = 5'd4,   // FSQRT.S
    FPU_FMADD  = 5'd5,   // FMADD.S
    FPU_FMSUB  = 5'd6,   // FMSUB.S
    FPU_FNMADD = 5'd7,   // FNMADD.S
    FPU_FNMSUB = 5'd8,   // FNMSUB.S
    FPU_CVT_WS = 5'd9,   // FCVT.W.S  (float → signed int)
    FPU_CVT_WUS= 5'd10,  // FCVT.WU.S (float → unsigned int)
    FPU_CVT_SW = 5'd11,  // FCVT.S.W  (signed int → float)
    FPU_CVT_SWU= 5'd12,  // FCVT.S.WU (unsigned int → float)
    FPU_EQ     = 5'd13,  // FEQ.S
    FPU_LT     = 5'd14,  // FLT.S
    FPU_LE     = 5'd15,  // FLE.S
    FPU_MIN    = 5'd16,  // FMIN.S
    FPU_MAX    = 5'd17,  // FMAX.S
    FPU_SGNJ   = 5'd18,  // FSGNJ.S
    FPU_SGNJN  = 5'd19,  // FSGNJN.S
    FPU_SGNJX  = 5'd20,  // FSGNJX.S
    FPU_CLASS  = 5'd21,  // FCLASS.S
    FPU_MVX2W  = 5'd22,  // FMV.X.W (float bits → int reg)
    FPU_MVW2X  = 5'd23   // FMV.W.X (int reg → float reg)
  } fpu_op_t;

  // ---------------------------------------------------------------------------
  // Forwarding Mux Select
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {
    FWD_NONE   = 2'b00,  // Use value from register file (no hazard)
    FWD_EX_MEM = 2'b01,  // Forward ALU result from EX/MEM pipeline register
    FWD_MEM_WB = 2'b10   // Forward result from MEM/WB pipeline register
  } fwd_sel_t;

  // ---------------------------------------------------------------------------
  // Write-Back Source Select
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {
    WB_ALU  = 2'b00,  // Write back ALU result
    WB_MEM  = 2'b01,  // Write back memory load data
    WB_PC4  = 2'b10,  // Write back PC+4 (JAL/JALR link address)
    WB_FPU  = 2'b11   // Write back FPU result
  } wb_sel_t;

  // ---------------------------------------------------------------------------
  // Immediate Type (for immediate generator)
  // ---------------------------------------------------------------------------
  typedef enum logic [2:0] {
    IMM_I = 3'b000,  // I-type: loads, JALR, arithmetic immediate
    IMM_S = 3'b001,  // S-type: stores
    IMM_B = 3'b010,  // B-type: branches
    IMM_U = 3'b011,  // U-type: LUI, AUIPC
    IMM_J = 3'b100,  // J-type: JAL
    IMM_Z = 3'b101   // Z-type: CSR zimm (zero-extended [19:15])
  } imm_type_t;

  // ---------------------------------------------------------------------------
  // funct3 Constants — Branch
  // ---------------------------------------------------------------------------
  localparam logic [2:0] F3_BEQ  = 3'b000;
  localparam logic [2:0] F3_BNE  = 3'b001;
  localparam logic [2:0] F3_BLT  = 3'b100;
  localparam logic [2:0] F3_BGE  = 3'b101;
  localparam logic [2:0] F3_BLTU = 3'b110;
  localparam logic [2:0] F3_BGEU = 3'b111;

  // ---------------------------------------------------------------------------
  // funct3 Constants — Load / Store
  // ---------------------------------------------------------------------------
  localparam logic [2:0] F3_LB  = 3'b000;
  localparam logic [2:0] F3_LH  = 3'b001;
  localparam logic [2:0] F3_LW  = 3'b010;
  localparam logic [2:0] F3_LBU = 3'b100;
  localparam logic [2:0] F3_LHU = 3'b101;
  localparam logic [2:0] F3_SB  = 3'b000;
  localparam logic [2:0] F3_SH  = 3'b001;
  localparam logic [2:0] F3_SW  = 3'b010;

  // ---------------------------------------------------------------------------
  // funct3 Constants — Integer ALU
  // ---------------------------------------------------------------------------
  localparam logic [2:0] F3_ADD_SUB = 3'b000;
  localparam logic [2:0] F3_SLL     = 3'b001;
  localparam logic [2:0] F3_SLT     = 3'b010;
  localparam logic [2:0] F3_SLTU    = 3'b011;
  localparam logic [2:0] F3_XOR     = 3'b100;
  localparam logic [2:0] F3_SRL_SRA = 3'b101;
  localparam logic [2:0] F3_OR      = 3'b110;
  localparam logic [2:0] F3_AND     = 3'b111;

  // ---------------------------------------------------------------------------
  // funct3 Constants — M Extension
  // ---------------------------------------------------------------------------
  localparam logic [2:0] F3_MUL    = 3'b000;
  localparam logic [2:0] F3_MULH   = 3'b001;
  localparam logic [2:0] F3_MULHSU = 3'b010;
  localparam logic [2:0] F3_MULHU  = 3'b011;
  localparam logic [2:0] F3_DIV    = 3'b100;
  localparam logic [2:0] F3_DIVU   = 3'b101;
  localparam logic [2:0] F3_REM    = 3'b110;
  localparam logic [2:0] F3_REMU   = 3'b111;

  // ---------------------------------------------------------------------------
  // funct3 Constants — CSR System Instructions
  // ---------------------------------------------------------------------------
  localparam logic [2:0] F3_ECALL  = 3'b000;  // also EBREAK / MRET
  localparam logic [2:0] F3_CSRRW  = 3'b001;
  localparam logic [2:0] F3_CSRRS  = 3'b010;
  localparam logic [2:0] F3_CSRRC  = 3'b011;
  localparam logic [2:0] F3_CSRRWI = 3'b101;
  localparam logic [2:0] F3_CSRRSI = 3'b110;
  localparam logic [2:0] F3_CSRRCI = 3'b111;

  // ---------------------------------------------------------------------------
  // funct7 Constants
  // ---------------------------------------------------------------------------
  localparam logic [6:0] F7_NORMAL = 7'b000_0000;  // ADD, SRL, etc.
  localparam logic [6:0] F7_ALT    = 7'b010_0000;  // SUB, SRA
  localparam logic [6:0] F7_MEXT   = 7'b000_0001;  // M-extension (MUL/DIV)

  // ---------------------------------------------------------------------------
  // Canonical NaN (IEEE 754 quiet NaN for RV32F)
  // ---------------------------------------------------------------------------
  localparam logic [31:0] CANONICAL_NAN = 32'h7FC0_0000;

  // ---------------------------------------------------------------------------
  // Reset / Boot Vector
  // ---------------------------------------------------------------------------
  localparam logic [31:0] RESET_VECTOR = 32'h0000_0000;

endpackage