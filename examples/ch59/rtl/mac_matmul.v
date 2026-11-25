// 제59장(선택): INT8 행렬-벡터 곱 MAC 블록 (y[r] = Σ W[r][c] · x[c], 32비트 누산)
//
// 구조: 2단 파이프라인
//   1단: 메모리에서 W[r][c], x[c]를 읽는다(동기 읽기, BRAM으로 합성되는 형태)
//   2단: 곱해서 누산기에 더한다. 행의 마지막 열이면 결과를 내보내고 누산기를 비운다
// 한 사이클에 곱셈 하나(MAC 1개). ROWS × COLS 사이클 + 파이프라인 2사이클이 걸린다.

module mac_matmul #(
    parameter ROWS   = 8,
    parameter COLS   = 896,
    parameter W_FILE = "w.hex",          // 행 우선, 한 줄에 INT8 하나(2의 보수 HEX)
    parameter X_FILE = "x.hex"
) (
    input  wire               clk,
    input  wire               rst,       // 동기 리셋
    input  wire               start,     // 한 사이클 펄스
    output reg                out_valid, // out_row, out_acc가 유효한 사이클
    output reg  [7:0]         out_row,
    output reg  signed [31:0] out_acc,
    output reg                done       // 마지막 행의 결과를 낸 뒤 1
);
    reg signed [7:0] w_mem [0:ROWS*COLS-1];
    reg signed [7:0] x_mem [0:COLS-1];
    initial begin
        $readmemh(W_FILE, w_mem);
        $readmemh(X_FILE, x_mem);
    end

    // 1단: 주소 생성과 메모리 읽기
    reg        running;
    reg [7:0]  row;
    reg [15:0] col;
    reg signed [7:0] w_q, x_q;
    reg        rd_valid, rd_last;
    reg [7:0]  rd_row;

    always @(posedge clk) begin
        if (rst) begin
            running  <= 1'b0;
            row      <= 8'd0;
            col      <= 16'd0;
            rd_valid <= 1'b0;
            rd_last  <= 1'b0;
        end else begin
            rd_valid <= 1'b0;
            rd_last  <= 1'b0;
            if (start && !running) begin
                running <= 1'b1;
                row     <= 8'd0;
                col     <= 16'd0;
            end else if (running) begin
                w_q      <= w_mem[row * COLS + col];
                x_q      <= x_mem[col];
                rd_valid <= 1'b1;
                rd_row   <= row;
                rd_last  <= (col == COLS - 1);
                if (col == COLS - 1) begin
                    col <= 16'd0;
                    if (row == ROWS - 1)
                        running <= 1'b0;
                    else
                        row <= row + 8'd1;
                end else begin
                    col <= col + 16'd1;
                end
            end
        end
    end

    // 2단: 곱셈과 누산(부호 있는 8비트 × 8비트 → 32비트로 부호 확장해 더한다)
    reg signed [31:0] acc;

    always @(posedge clk) begin
        if (rst) begin
            acc       <= 32'sd0;
            out_valid <= 1'b0;
            done      <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            if (start)
                done <= 1'b0;
            if (rd_valid) begin
                if (rd_last) begin
                    out_acc   <= acc + w_q * x_q;
                    out_row   <= rd_row;
                    out_valid <= 1'b1;
                    acc       <= 32'sd0;
                    if (rd_row == ROWS - 1)
                        done <= 1'b1;
                end else begin
                    acc <= acc + w_q * x_q;
                end
            end
        end
    end
endmodule
