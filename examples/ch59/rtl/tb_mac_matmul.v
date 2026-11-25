// 제59장(선택): mac_matmul Testbench — Python 정수 기준값(expected.hex)과 비트 단위로 비교한다
//
// 실행 예(Icarus Verilog, 교재 검증 환경에서는 미검증):
//   iverilog -o tb tb_mac_matmul.v mac_matmul.v && vvp tb

`timescale 1ns / 1ps

module tb_mac_matmul;
    localparam ROWS = 8, COLS = 896;

    reg clk = 1'b0, rst = 1'b1, start = 1'b0;
    wire out_valid, done;
    wire [7:0] out_row;
    wire signed [31:0] out_acc;

    reg [31:0] expected [0:ROWS-1];
    integer errors = 0, received = 0, cycles = 0;

    mac_matmul #(.ROWS(ROWS), .COLS(COLS)) dut (
        .clk(clk), .rst(rst), .start(start),
        .out_valid(out_valid), .out_row(out_row), .out_acc(out_acc), .done(done)
    );

    always #5 clk = ~clk;                         // 100MHz

    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (out_valid) begin
            received <= received + 1;
            if (out_acc !== $signed(expected[out_row])) begin
                errors <= errors + 1;
                $display("FAIL row %0d: got %0d, expected %0d", out_row, out_acc, $signed(expected[out_row]));
            end else begin
                $display("ok   row %0d: %0d", out_row, out_acc);
            end
        end
    end

    initial begin
        $readmemh("expected.hex", expected);
        repeat (3) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk) start <= 1'b1;
        @(posedge clk) start <= 1'b0;
        wait (done);
        @(posedge clk);
        $display("%s: %0d rows, %0d errors, %0d cycles", (errors == 0 && received == ROWS) ? "PASS" : "FAIL",
                 received, errors, cycles);
        $finish;
    end

    initial begin
        #(20 * ROWS * COLS * 10);                  // 시간 초과 보호
        $display("FAIL: timeout");
        $finish;
    end
endmodule
