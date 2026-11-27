// Interpolacion por tramos del ADC filtrado. Los ocho valores se miden
// con la alimentacion y geometria definitivas y se generan en ir_calibration.vh.
`include "src/ir_calibration.vh"

module ir_adc_to_mm #(
    parameter integer SENSOR = 1
) (
    input wire unstable,
    input wire [9:0] adc,
    output reg [2:0] status,
    output reg [15:0] distance_mm
);
    localparam integer A12 = SENSOR == 1 ? `IR1_ADC_12 : `IR2_ADC_12;
    localparam integer A20 = SENSOR == 1 ? `IR1_ADC_20 : `IR2_ADC_20;
    localparam integer A30 = SENSOR == 1 ? `IR1_ADC_30 : `IR2_ADC_30;
    localparam integer A40 = SENSOR == 1 ? `IR1_ADC_40 : `IR2_ADC_40;
    localparam integer A50 = SENSOR == 1 ? `IR1_ADC_50 : `IR2_ADC_50;
    localparam integer A60 = SENSOR == 1 ? `IR1_ADC_60 : `IR2_ADC_60;
    localparam integer A70 = SENSOR == 1 ? `IR1_ADC_70 : `IR2_ADC_70;
    localparam integer A80 = SENSOR == 1 ? `IR1_ADC_80 : `IR2_ADC_80;
    localparam integer TABLE_VALID = (SENSOR == 1 ? `IR1_CALIBRATED : `IR2_CALIBRATED) &&
        A12 <= 1023 && A12 > A20 && A20 > A30 && A30 > A40 &&
        A40 > A50 && A50 > A60 && A60 > A70 && A70 > A80 && A80 > 10;
    // Denominadores nunca nulos, incluso con la tabla aun vacia.
    localparam integer D12 = A12 > A20 ? A12 - A20 : 1;
    localparam integer D20 = A20 > A30 ? A20 - A30 : 1;
    localparam integer D30 = A30 > A40 ? A30 - A40 : 1;
    localparam integer D40 = A40 > A50 ? A40 - A50 : 1;
    localparam integer D50 = A50 > A60 ? A50 - A60 : 1;
    localparam integer D60 = A60 > A70 ? A60 - A70 : 1;
    localparam integer D70 = A70 > A80 ? A70 - A80 : 1;

    // Ganancias Q10 constantes. Seleccionar tramo ANTES de multiplicar
    // permite compartir un solo multiplicador por sensor.
    localparam integer G12 = (80*1024 + D12/2) / D12;
    localparam integer G20 = (100*1024 + D20/2) / D20;
    localparam integer G30 = (100*1024 + D30/2) / D30;
    localparam integer G40 = (100*1024 + D40/2) / D40;
    localparam integer G50 = (100*1024 + D50/2) / D50;
    localparam integer G60 = (100*1024 + D60/2) / D60;
    localparam integer G70 = (100*1024 + D70/2) / D70;
    reg [9:0] delta_adc;
    reg [16:0] gain_q10;
    reg [15:0] base_mm;
    wire [26:0] scaled_round = delta_adc * gain_q10 + 27'd512;

    always @* begin
        delta_adc = 0;
        gain_q10 = 0;
        base_mm = 0;
        if (adc >= A20) begin
            delta_adc = A12 - adc;
            gain_q10 = G12;
            base_mm = 120;
        end else if (adc >= A30) begin
            delta_adc = A20 - adc;
            gain_q10 = G20;
            base_mm = 200;
        end else if (adc >= A40) begin
            delta_adc = A30 - adc;
            gain_q10 = G30;
            base_mm = 300;
        end else if (adc >= A50) begin
            delta_adc = A40 - adc;
            gain_q10 = G40;
            base_mm = 400;
        end else if (adc >= A60) begin
            delta_adc = A50 - adc;
            gain_q10 = G50;
            base_mm = 500;
        end else if (adc >= A70) begin
            delta_adc = A60 - adc;
            gain_q10 = G60;
            base_mm = 600;
        end else begin
            delta_adc = A70 - adc;
            gain_q10 = G70;
            base_mm = 700;
        end
    end

    // Estados: 0=OK, 1=limite 120, 2=limite 800, 3=menor de 120,
    // 4=mayor de 800, 5=ADC inestable, 6=tabla sin calibrar.
    always @* begin
        status = 3'd6;
        distance_mm = 16'hFFFF;
        if (unstable) begin
            status = 3'd5;
        end else if (TABLE_VALID) begin
            if (adc > A12) begin
                if (adc <= A12 + 12) begin
                    status = 3'd1;
                    distance_mm = 16'd120;
                end else status = 3'd3;
            end else if (adc < A80) begin
                if (adc >= A80 - 10) begin
                    status = 3'd2;
                    distance_mm = 16'd800;
                end else status = 3'd4;
            end else begin
                status = 3'd0;
                distance_mm = base_mm + scaled_round[25:10];
            end
        end
    end
endmodule
