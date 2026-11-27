// K: signed span(8), margin(4), stop tolerance(4), B window(4), XOR(2), LF.
// G: unsigned goal(8), arrival window(4), XOR(2), LF.
// A: signed absolute B(8); I: session token(8).
// Other commands are one ASCII letter followed by LF.
module command_parser (
    input wire clk,
    input wire [7:0] data,
    input wire data_valid,
    output reg [7:0] command,
    output reg command_valid,
    output reg signed [31:0] config_span,
    output reg [15:0] config_margin,
    output reg [15:0] config_tolerance,
    output reg [15:0] config_bottom_window,
    output reg signed [31:0] absolute_bottom,
    output reg [31:0] session_token,
    output reg [31:0] goal,
    output reg [15:0] goal_window
);
    initial begin
        command = 0;
        command_valid = 0;
        config_span = 0;
        config_margin = 0;
        config_tolerance = 0;
        config_bottom_window = 0;
        absolute_bottom = 0;
        session_token = 0;
        goal = 0;
        goal_window = 0;
    end
    reg [7:0] first = 0;
    reg [5:0] index = 0;
    reg bad = 0;
    reg [7:0] xor_sum = 0;
    reg [7:0] received_sum = 0;

    function [4:0] hex_value;
        input [7:0] ascii;
        begin
            if (ascii >= "0" && ascii <= "9") hex_value = {1'b0, ascii[3:0]};
            else if (ascii >= "A" && ascii <= "F") hex_value = {1'b0, ascii[3:0] + 4'd9};
            else hex_value = 5'b10000;
        end
    endfunction

    wire [4:0] digit = hex_value(data);
    always @(posedge clk) begin
        command_valid <= 0;
        if (data_valid) begin
            if (data == 8'h0A) begin
                if (!bad && ((index == 1 &&
                    (first == "M" || first == "D" || first == "Z" ||
                     first == "H" || first == "W" || first == "S" || first == "X")) ||
                    (first == "K" && index == 23 && xor_sum == received_sum) ||
                    (first == "G" && index == 15 && xor_sum == received_sum) ||
                    ((first == "A" || first == "I") &&
                     index == 11 && xor_sum == received_sum))) begin
                    command <= first;
                    command_valid <= 1;
                end
                index <= 0;
                bad <= 0;
                xor_sum <= 0;
                received_sum <= 0;
            end else if (index == 0) begin
                first <= data;
                index <= 1;
                xor_sum <= data;
                bad <= 0;
                config_span <= 0;
                config_margin <= 0;
                config_tolerance <= 0;
                config_bottom_window <= 0;
                absolute_bottom <= 0;
                session_token <= 0;
                goal <= 0;
                goal_window <= 0;
            end else if (index < 31) begin
                index <= index + 1'b1;
                if (first == "K" && index >= 1 && index <= 20) begin
                    if (digit[4]) bad <= 1;
                    else if (index <= 8) config_span <= {config_span[27:0], digit[3:0]};
                    else if (index <= 12) config_margin <= {config_margin[11:0], digit[3:0]};
                    else if (index <= 16) config_tolerance <= {config_tolerance[11:0], digit[3:0]};
                    else config_bottom_window <= {config_bottom_window[11:0], digit[3:0]};
                    xor_sum <= xor_sum ^ data;
                end else if (first == "G" && index >= 1 && index <= 12) begin
                    if (digit[4]) bad <= 1;
                    else if (index <= 8) goal <= {goal[27:0], digit[3:0]};
                    else goal_window <= {goal_window[11:0], digit[3:0]};
                    xor_sum <= xor_sum ^ data;
                end else if ((first == "A" || first == "I") &&
                             index >= 1 && index <= 8) begin
                    if (digit[4]) bad <= 1;
                    else if (first == "A") absolute_bottom <= {absolute_bottom[27:0], digit[3:0]};
                    else session_token <= {session_token[27:0], digit[3:0]};
                    xor_sum <= xor_sum ^ data;
                end else if ((first == "K" && (index == 21 || index == 22)) ||
                             (first == "G" && (index == 13 || index == 14)) ||
                             ((first == "A" || first == "I") &&
                              (index == 9 || index == 10))) begin
                    if (digit[4]) bad <= 1;
                    else received_sum <= {received_sum[3:0], digit[3:0]};
                end else bad <= 1;
            end else bad <= 1;
        end
    end
endmodule
