// Top module name and pin names retained to reuse the proven Tang Nano 9K CST.
module MotorTest #(
    parameter integer CLK_HZ = 27000000,
    parameter integer BAUD = 115200
)(
    input wire clk,
    input wire enc_a,
    input wire enc_b,
    input wire uart_rx_pin,
    input wire ir_rx_pin,
    output wire usb_uart_tx,
    output wire in3, // pin 25 -> L298N IN1, down
    output wire in4, // pin 26 -> L298N IN2, up
    output wire enb, // pin 27 -> L298N ENA
    output wire [5:0] led
);
    localparam integer PWM_PERIOD = CLK_HZ / 10000;
    localparam integer DUTY_70 = PWM_PERIOD * 70 / 100;
    localparam integer DUTY_71 = PWM_PERIOD * 71 / 100;
    localparam integer DUTY_72 = PWM_PERIOD * 72 / 100;
    localparam integer DUTY_73 = PWM_PERIOD * 73 / 100;
    localparam integer DUTY_74 = PWM_PERIOD * 74 / 100;
    localparam integer DUTY_75 = PWM_PERIOD * 75 / 100;
    localparam integer TELEMETRY_CYCLES = CLK_HZ / 5;

    wire signed [31:0] position;
    wire encoder_direction;
    quadrature_decoder encoder (
        .clk(clk), .enc_a(enc_a), .enc_b(enc_b),
        .position(position), .direction(encoder_direction)
    );

    wire [7:0] uart_data;
    wire uart_valid;
    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) receiver (
        .clk(clk), .rx(uart_rx_pin), .data(uart_data), .valid(uart_valid)
    );
    wire ir_fresh;
    wire [2:0] ir_status1, ir_status2;
    wire [15:0] ir_distance_mm1, ir_adc1, ir_spread1;
    wire [15:0] ir_distance_mm2, ir_adc2, ir_spread2;
    ir_packet_decoder #(.CLK_HZ(CLK_HZ), .BAUD(9600)) ir_receiver (
        .clk(clk), .rx(ir_rx_pin), .fresh(ir_fresh),
        .status1(ir_status1), .distance_mm1(ir_distance_mm1),
        .adc1(ir_adc1), .spread1(ir_spread1),
        .status2(ir_status2), .distance_mm2(ir_distance_mm2),
        .adc2(ir_adc2), .spread2(ir_spread2)
    );
    wire [7:0] command;
    wire command_valid;
    wire signed [31:0] config_span;
    wire [15:0] config_margin, config_tolerance, config_bottom_window;
    wire [31:0] config_goal;
    wire [15:0] config_goal_window;
    wire signed [31:0] absolute_bottom;
    wire [31:0] config_session_token, session_token;
    command_parser parser (
        .clk(clk), .data(uart_data), .data_valid(uart_valid),
        .command(command), .command_valid(command_valid),
        .config_span(config_span), .config_margin(config_margin),
        .config_tolerance(config_tolerance), .config_bottom_window(config_bottom_window),
        .absolute_bottom(absolute_bottom), .session_token(config_session_token),
        .goal(config_goal), .goal_window(config_goal_window)
    );

    wire [2:0] mode;
    wire [1:0] drive;
    wire motor_up, motor_down, moving, busy;
    wire [3:0] fault_code;
    wire [6:0] pwm_percent;
    wire signed [31:0] span, relative;
    wire [31:0] goal;
    position_controller #(.CLK_HZ(CLK_HZ)) controller (
        .clk(clk), .position(position), .command(command),
        .command_valid(command_valid), .config_span(config_span),
        .config_margin(config_margin), .config_tolerance(config_tolerance),
        .config_bottom_window(config_bottom_window), .absolute_bottom(absolute_bottom),
        .config_session_token(config_session_token), .session_token(session_token),
        .config_goal(config_goal), .config_goal_window(config_goal_window),
        .mode(mode), .drive(drive),
        .motor_up(motor_up), .motor_down(motor_down),
        .span(span), .goal(goal), .relative(relative),
        .margin(), .moving(moving), .busy(busy), .pwm_percent(pwm_percent),
        .fault_code(fault_code)
    );

    reg [31:0] pwm_counter = 0;
    always @(posedge clk) begin
        if (pwm_counter >= PWM_PERIOD - 1) pwm_counter <= 0;
        else pwm_counter <= pwm_counter + 1'b1;
    end
    assign in3 = motor_down;
    assign in4 = motor_up;
    wire [31:0] pwm_duty = pwm_percent == 70 ? DUTY_70 :
                           pwm_percent == 71 ? DUTY_71 :
                           pwm_percent == 72 ? DUTY_72 :
                           pwm_percent == 73 ? DUTY_73 :
                           pwm_percent == 74 ? DUTY_74 : DUTY_75;
    assign enb = moving && (pwm_counter < pwm_duty);

    reg [31:0] blink_timer = 0;
    reg heartbeat_led = 1;
    always @(posedge clk) begin
        if (blink_timer >= CLK_HZ/2 - 1) begin
            blink_timer <= 0;
            heartbeat_led <= ~heartbeat_led;
        end else blink_timer <= blink_timer + 1'b1;
    end
    assign led = {~motor_down, ~motor_up, mode != 3'd4,
                  mode != 3'd3, mode != 3'd1, heartbeat_led};

    reg [7:0] reply = 0;
    reg reply_send = 0;
    wire reply_busy;
    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) transmitter (
        .clk(clk), .send(reply_send), .data(reply),
        .tx(usb_uart_tx), .busy(reply_busy)
    );
    reg [31:0] telemetry_timer = 0;
    reg telemetry_active = 0;
    reg [6:0] telemetry_index = 0;
    reg [31:0] raw_snapshot = 0, relative_snapshot = 0;
    reg [31:0] span_snapshot = 0, goal_snapshot = 0;
    reg [31:0] token_snapshot = 0;
    reg [2:0] mode_snapshot = 0;
    reg moving_snapshot = 0, busy_snapshot = 0;
    reg [3:0] fault_snapshot = 0;
    reg ir_fresh_snapshot = 0;
    reg [2:0] ir_status1_snapshot = 0, ir_status2_snapshot = 0;
    reg [15:0] ir_mm1_snapshot = 16'hFFFF, ir_adc1_snapshot = 16'hFFFF;
    reg [15:0] ir_spread1_snapshot = 16'hFFFF;
    reg [15:0] ir_mm2_snapshot = 16'hFFFF, ir_adc2_snapshot = 16'hFFFF;
    reg [15:0] ir_spread2_snapshot = 16'hFFFF;

    function [7:0] hex_digit;
        input [3:0] value;
        begin
            if (value < 10) hex_digit = "0" + value;
            else hex_digit = "A" + value - 4'd10;
        end
    endfunction
    function [7:0] hex_at16;
        input [15:0] value;
        input [1:0] index;
        reg [15:0] shifted;
        begin
            shifted = value >> ((3 - index) * 4);
            hex_at16 = hex_digit(shifted[3:0]);
        end
    endfunction
    function [7:0] hex_at32;
        input [31:0] value;
        input [2:0] index;
        reg [31:0] shifted;
        begin
            shifted = value >> ((7 - index) * 4);
            hex_at32 = hex_digit(shifted[3:0]);
        end
    endfunction
    function [7:0] mode_char;
        input [2:0] value;
        begin
            case (value)
                3'd1: mode_char = "M";
                3'd2: mode_char = "C";
                3'd3: mode_char = "R";
                3'd4: mode_char = "F";
                default: mode_char = "D";
            endcase
        end
    endfunction

    // @raw8 LF, #relative8 LF, !mode motor busy fault span8 goal8 LF,
    // $token8 LF, ?estado1 mm1 adc1 LF, ^dispersion1 LF,
    // &estado2 mm2 adc2 LF, %dispersion2 LF.
    // F = lectura IR vencida; 5 = inestable; 6 = sin calibrar.
    always @(posedge clk) begin
        reply_send <= 0;
        if (telemetry_timer >= TELEMETRY_CYCLES - 1) begin
            telemetry_timer <= 0;
            if (!telemetry_active) begin
                raw_snapshot <= position;
                relative_snapshot <= relative;
                span_snapshot <= span;
                goal_snapshot <= goal;
                token_snapshot <= session_token;
                mode_snapshot <= mode;
                moving_snapshot <= moving;
                busy_snapshot <= busy;
                fault_snapshot <= fault_code;
                ir_fresh_snapshot <= ir_fresh;
                ir_status1_snapshot <= ir_status1;
                ir_mm1_snapshot <= ir_fresh ? ir_distance_mm1 : 16'hFFFF;
                ir_adc1_snapshot <= ir_fresh ? ir_adc1 : 16'hFFFF;
                ir_spread1_snapshot <= ir_fresh ? ir_spread1 : 16'hFFFF;
                ir_status2_snapshot <= ir_status2;
                ir_mm2_snapshot <= ir_fresh ? ir_distance_mm2 : 16'hFFFF;
                ir_adc2_snapshot <= ir_fresh ? ir_adc2 : 16'hFFFF;
                ir_spread2_snapshot <= ir_fresh ? ir_spread2 : 16'hFFFF;
                telemetry_index <= 0;
                telemetry_active <= 1;
            end
        end else telemetry_timer <= telemetry_timer + 1'b1;

        if (telemetry_active && !reply_busy && !reply_send) begin
            if (telemetry_index == 0) reply <= "@";
            else if (telemetry_index <= 8)
                reply <= hex_at32(raw_snapshot, telemetry_index - 1'b1);
            else if (telemetry_index == 9) reply <= 8'h0A;
            else if (telemetry_index == 10) reply <= "#";
            else if (telemetry_index <= 18)
                reply <= hex_at32(relative_snapshot, telemetry_index - 6'd11);
            else if (telemetry_index == 19) reply <= 8'h0A;
            else if (telemetry_index == 20) reply <= "!";
            else if (telemetry_index == 21) reply <= mode_char(mode_snapshot);
            else if (telemetry_index == 22) reply <= moving_snapshot ? "1" : "0";
            else if (telemetry_index == 23) reply <= busy_snapshot ? "1" : "0";
            else if (telemetry_index == 24) reply <= hex_digit(fault_snapshot);
            else if (telemetry_index <= 32)
                reply <= hex_at32(span_snapshot, telemetry_index - 6'd25);
            else if (telemetry_index <= 40)
                reply <= hex_at32(goal_snapshot, telemetry_index - 6'd33);
            else if (telemetry_index == 41) reply <= 8'h0A;
            else if (telemetry_index == 42) reply <= "$";
            else if (telemetry_index <= 50)
                reply <= hex_at32(token_snapshot, telemetry_index - 6'd43);
            else if (telemetry_index == 51) reply <= 8'h0A;
            else if (telemetry_index == 52) reply <= "?";
            else if (telemetry_index == 53)
                reply <= ir_fresh_snapshot ? hex_digit({1'b0, ir_status1_snapshot}) : "F";
            else if (telemetry_index <= 57)
                reply <= hex_at16(ir_mm1_snapshot, telemetry_index - 6'd54);
            else if (telemetry_index <= 61)
                reply <= hex_at16(ir_adc1_snapshot, telemetry_index - 6'd58);
            else if (telemetry_index == 62) reply <= 8'h0A;
            else if (telemetry_index == 63) reply <= "^";
            else if (telemetry_index <= 67)
                reply <= hex_at16(ir_spread1_snapshot, telemetry_index - 7'd64);
            else if (telemetry_index == 68) reply <= 8'h0A;
            else if (telemetry_index == 69) reply <= "&";
            else if (telemetry_index == 70)
                reply <= ir_fresh_snapshot ? hex_digit({1'b0, ir_status2_snapshot}) : "F";
            else if (telemetry_index <= 74)
                reply <= hex_at16(ir_mm2_snapshot, telemetry_index - 7'd71);
            else if (telemetry_index <= 78)
                reply <= hex_at16(ir_adc2_snapshot, telemetry_index - 7'd75);
            else if (telemetry_index == 79) reply <= 8'h0A;
            else if (telemetry_index == 80) reply <= "%";
            else if (telemetry_index <= 84)
                reply <= hex_at16(ir_spread2_snapshot, telemetry_index - 7'd81);
            else reply <= 8'h0A;
            reply_send <= 1;
            if (telemetry_index == 85) telemetry_active <= 0;
            else telemetry_index <= telemetry_index + 1'b1;
        end
    end
endmodule
