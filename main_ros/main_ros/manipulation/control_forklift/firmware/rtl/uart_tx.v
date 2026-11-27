// One-byte UART status reply over the Tang Nano 9K onboard USB-UART.
module uart_tx #(
    parameter integer CLK_HZ = 27000000,
    parameter integer BAUD = 115200
)(
    input wire clk,
    input wire send,
    input wire [7:0] data,
    output reg tx,
    output wire busy
);
    initial tx = 1;
    localparam integer BIT_CYCLES = (CLK_HZ + BAUD/2) / BAUD;
    localparam [1:0] IDLE = 0, START = 1, BITS = 2, STOP = 3;
    reg [1:0] state = IDLE;
    reg [31:0] timer = 0;
    reg [2:0] bit_index = 0;
    reg [7:0] shift = 0;
    assign busy = state != IDLE;

    always @(posedge clk) begin
        if (state == IDLE) begin
            tx <= 1;
            if (send) begin
                shift <= data;
                bit_index <= 0;
                timer <= BIT_CYCLES - 1;
                tx <= 0;
                state <= START;
            end
        end else if (timer != 0) begin
            timer <= timer - 1'b1;
        end else begin
            timer <= BIT_CYCLES - 1;
            case (state)
                START: begin
                    tx <= shift[0];
                    state <= BITS;
                end
                BITS: if (bit_index == 7) begin
                    tx <= 1;
                    state <= STOP;
                end else begin
                    bit_index <= bit_index + 1'b1;
                    tx <= shift[bit_index + 1'b1];
                end
                STOP: state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
