// Nano Every Serial1: 9600 8N1.
// A9 q1 adc1_lo adc1_hi spread1_lo spread1_hi
//    q2 adc2_lo adc2_hi spread2_lo spread2_hi XOR.
// Los dos sensores comparten UART, pero sus estados y tablas son independientes.
module ir_packet_decoder #(
    parameter integer CLK_HZ = 27000000,
    parameter integer BAUD = 9600
)(
    input wire clk,
    input wire rx,
    output reg fresh = 0,
    output reg [2:0] status1 = 3'd6,
    output reg [15:0] distance_mm1 = 16'hFFFF,
    output reg [15:0] adc1 = 16'hFFFF,
    output reg [15:0] spread1 = 16'hFFFF,
    output reg [2:0] status2 = 3'd6,
    output reg [15:0] distance_mm2 = 16'hFFFF,
    output reg [15:0] adc2 = 16'hFFFF,
    output reg [15:0] spread2 = 16'hFFFF
);
    localparam integer STALE_CYCLES = CLK_HZ * 2;
    wire [7:0] data;
    wire data_valid;
    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) receiver (
        .clk(clk), .rx(rx), .data(data), .valid(data_valid)
    );

    reg [3:0] index = 0;
    reg [7:0] xor_value = 0;
    reg [7:0] quality1 = 0, quality2 = 0;
    reg [7:0] adc1_lo = 0, adc1_hi = 0, adc2_lo = 0, adc2_hi = 0;
    reg [7:0] spread1_lo = 0, spread1_hi = 0, spread2_lo = 0, spread2_hi = 0;
    reg [31:0] age = 0;
    wire [15:0] packet_adc1 = {adc1_hi, adc1_lo};
    wire [15:0] packet_adc2 = {adc2_hi, adc2_lo};
    wire [15:0] packet_spread1 = {spread1_hi, spread1_lo};
    wire [15:0] packet_spread2 = {spread2_hi, spread2_lo};
    wire [2:0] converted_status1, converted_status2;
    wire [15:0] converted_mm1, converted_mm2;
    ir_adc_to_mm #(.SENSOR(1)) converter1 (
        .unstable(quality1 != 0), .adc(packet_adc1[9:0]),
        .status(converted_status1), .distance_mm(converted_mm1)
    );
    ir_adc_to_mm #(.SENSOR(2)) converter2 (
        .unstable(quality2 != 0), .adc(packet_adc2[9:0]),
        .status(converted_status2), .distance_mm(converted_mm2)
    );
    wire fields_valid =
        quality1 <= 1 && packet_adc1 <= 1023 &&
        (packet_spread1 <= 1023 || (quality1 == 1 && packet_spread1 == 16'hFFFF)) &&
        quality2 <= 1 && packet_adc2 <= 1023 &&
        (packet_spread2 <= 1023 || (quality2 == 1 && packet_spread2 == 16'hFFFF));

    always @(posedge clk) begin
        if (age < STALE_CYCLES) age <= age + 1'b1;
        else fresh <= 0;

        if (data_valid) begin
            case (index)
                0: if (data == 8'hA9) begin
                    xor_value <= 8'hA9;
                    index <= 1;
                end
                1: begin quality1 <= data; xor_value <= xor_value ^ data; index <= 2; end
                2: begin adc1_lo <= data; xor_value <= xor_value ^ data; index <= 3; end
                3: begin adc1_hi <= data; xor_value <= xor_value ^ data; index <= 4; end
                4: begin spread1_lo <= data; xor_value <= xor_value ^ data; index <= 5; end
                5: begin spread1_hi <= data; xor_value <= xor_value ^ data; index <= 6; end
                6: begin quality2 <= data; xor_value <= xor_value ^ data; index <= 7; end
                7: begin adc2_lo <= data; xor_value <= xor_value ^ data; index <= 8; end
                8: begin adc2_hi <= data; xor_value <= xor_value ^ data; index <= 9; end
                9: begin spread2_lo <= data; xor_value <= xor_value ^ data; index <= 10; end
                10: begin spread2_hi <= data; xor_value <= xor_value ^ data; index <= 11; end
                11: begin
                    if (data == xor_value && fields_valid) begin
                        status1 <= converted_status1;
                        distance_mm1 <= converted_mm1;
                        adc1 <= packet_adc1;
                        spread1 <= packet_spread1;
                        status2 <= converted_status2;
                        distance_mm2 <= converted_mm2;
                        adc2 <= packet_adc2;
                        spread2 <= packet_spread2;
                        age <= 0;
                        fresh <= 1;
                        index <= 0;
                    end else if (data == 8'hA9) begin
                        xor_value <= 8'hA9;
                        index <= 1;
                    end else index <= 0;
                end
                default: index <= 0;
            endcase
        end
    end
endmodule
