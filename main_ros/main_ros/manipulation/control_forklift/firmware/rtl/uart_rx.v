// UART receiver for the Tang Nano 9K onboard USB-UART bridge.
module uart_rx #(
    parameter integer CLK_HZ = 27000000,
    parameter integer BAUD = 115200
)(
    input wire clk,
    input wire rx,
    output reg [7:0] data,
    output reg valid
);
    initial begin
        data = 0;
        valid = 0;
    end
    localparam integer BIT_CYCLES = (CLK_HZ + BAUD/2) / BAUD;
    reg rx_meta = 1, rx_sync = 1;
    reg [1:0] state = 0;
    reg [31:0] timer = 0;
    reg [2:0] bit_index = 0;
    reg [7:0] shift = 0;

    always @(posedge clk) begin
        rx_meta <= rx;
        rx_sync <= rx_meta;
        valid <= 0;
        case (state)
            0: if (!rx_sync) begin
                timer <= BIT_CYCLES/2 - 1;
                state <= 1;
            end
            1: if (timer != 0) timer <= timer - 1;
               else if (!rx_sync) begin
                   timer <= BIT_CYCLES - 1;
                   bit_index <= 0;
                   state <= 2;
               end else state <= 0;
            2: if (timer != 0) timer <= timer - 1;
               else begin
                   shift[bit_index] <= rx_sync;
                   timer <= BIT_CYCLES - 1;
                   if (bit_index == 7) state <= 3;
                   else bit_index <= bit_index + 1'b1;
               end
            3: if (timer != 0) timer <= timer - 1;
               else begin
                   if (rx_sync) begin
                       data <= shift;
                       valid <= 1;
                   end
                   state <= 0;
               end
        endcase
    end
endmodule
