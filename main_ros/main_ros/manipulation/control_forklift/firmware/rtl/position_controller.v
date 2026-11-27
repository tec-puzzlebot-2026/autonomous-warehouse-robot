// Volatile controller. A new power cycle always starts DISARMED.
// Up is normalized positive regardless of the encoder's electrical sign.
module position_controller #(
    parameter integer CLK_HZ = 27000000,
    parameter integer WATCHDOG_MS = 300,
    parameter integer STABLE_MS = 600,
    parameter integer STALL_MS = 500,
    parameter integer DEAD_MS = 20,
    parameter integer PROGRESS_MS = 15000,
    parameter integer PULSE_STEP_MS = 20
)(
    input wire clk,
    input wire signed [31:0] position,
    input wire [7:0] command,
    input wire command_valid,
    input wire signed [31:0] config_span,
    input wire [15:0] config_margin,
    input wire [15:0] config_tolerance,
    input wire [15:0] config_bottom_window,
    input wire signed [31:0] absolute_bottom,
    input wire [31:0] config_session_token,
    input wire [31:0] config_goal,
    input wire [15:0] config_goal_window,
    output reg [2:0] mode, // D=0, M=1, C=2, R=3, F=4
    output reg [1:0] drive, // stop=0, up=1, down=2
    output wire motor_up,
    output wire motor_down,
    output reg signed [31:0] span,
    output reg [31:0] goal,
    output wire signed [31:0] relative,
    output reg [15:0] margin,
    output reg [31:0] session_token,
    output wire moving,
    output wire busy,
    output reg [6:0] pwm_percent,
    output reg [3:0] fault_code
);
    localparam [2:0] DISARMED=0, MANUAL=1, CONFIGURED=2, READY=3, FAULT=4;
    localparam [1:0] STOP=0, UP=1, DOWN=2;
    initial begin
        mode = DISARMED;
        drive = STOP;
        span = 0;
        goal = 0;
        margin = 0;
        session_token = 0;
        pwm_percent = 75;
        fault_code = 0;
    end
    localparam integer WATCHDOG_CYCLES=(CLK_HZ/1000)*WATCHDOG_MS;
    localparam integer STABLE_CYCLES=(CLK_HZ/1000)*STABLE_MS;
    localparam integer STALL_CYCLES=(CLK_HZ/1000)*STALL_MS;
    localparam integer DEAD_CYCLES=(CLK_HZ/1000)*DEAD_MS;
    localparam integer PROGRESS_CYCLES=(CLK_HZ/1000)*PROGRESS_MS;
    localparam integer PULSE_STEP_CYCLES=(CLK_HZ/1000)*PULSE_STEP_MS;
    localparam integer SPEED_SAMPLE_CYCLES=CLK_HZ/50;

    reg signed [31:0] reference = 0;
    reg [15:0] tolerance = 0;
    reg [15:0] bottom_window = 0;
    reg [15:0] arrival_window = 0;
    reg goal_active = 0;
    reg [31:0] watchdog = 0, jog_timer = 0, stable_timer = 0;
    reg [31:0] stall_timer = 0, progress_timer = 0, dead_timer = 0;
    reg [31:0] pulse_timer = 0;
    reg signed [31:0] previous_position = 0;
    reg position_changed_last_cycle = 0;
    reg signed [32:0] drive_origin = 0;
    reg signed [32:0] sampled_rel = 0, velocity = 0;
    reg [31:0] speed_timer = 0;
    reg settling = 0;
    reg hold_active = 0;
    reg hold_correction = 0;
    reg pulse_active = 0;
    reg last_pulsed = 0;
    reg [1:0] last_drive = STOP;
    reg [3:0] lead_up = 4'd5, lead_down = 4'd7;
    reg [3:0] pulse_up_steps = 4'd4, pulse_down_steps = 4'd4;
    reg signed [32:0] best_excess = 0;
    reg signed [32:0] settled_excess = 0, settled_rel = 0;
    reg compute_excess = 0;
    reg evaluate_settled = 0;
    reg progress_started = 0;
    wire [31:0] abs_span = span[31] ? -span : span;
    wire [31:0] cfg_abs_span = config_span[31] ? -config_span : config_span;
    wire [31:0] max_goal = abs_span - margin;
    wire signed [32:0] raw_diff = $signed({position[31],position}) - $signed({reference[31],reference});
    wire signed [32:0] rel = span[31] ? -raw_diff : raw_diff;
    // Las decisiones de control usan una lectura registrada; los topes del
    // motor siguen usando rel inmediato para no retrasar la protección.
    reg signed [32:0] control_rel = 0;
    wire signed [32:0] goal_ext = $signed({1'b0,goal});
    wire signed [32:0] max_ext = $signed({1'b0,max_goal});
    wire signed [32:0] safety_max_ext = $signed({1'b0,abs_span}) + bottom_window_ext;
    wire signed [32:0] bottom_window_ext = $signed({1'b0,16'd0,bottom_window});
    wire signed [32:0] arrival_window_ext = $signed({1'b0,16'd0,arrival_window});
    wire signed [32:0] error_counts = goal_ext - control_rel;
    wire signed [32:0] abs_error = error_counts < 0 ? -error_counts : error_counts;
    wire signed [32:0] settled_error = goal_ext - settled_rel;
    wire signed [32:0] settled_abs_error = settled_error < 0 ?
                                           -settled_error : settled_error;
    wire signed [32:0] measured_excess = settled_abs_error > arrival_window_ext ?
                                         settled_abs_error - arrival_window_ext : 33'sd0;
    wire meaningful_progress = best_excess >= settled_excess + 33'sd3;
    // La velocidad se mide en cuentas por 20 ms. El adelanto se adapta por
    // dirección y apunta al centro de la meta, no al borde de la ventana.
    wire signed [35:0] velocity_wide = {{3{velocity[32]}},velocity};
    wire signed [35:0] rel_wide = {{3{control_rel[32]}},control_rel};
    wire signed [35:0] up_advance =
        (lead_up[0] ? velocity_wide : 36'sd0) +
        (lead_up[1] ? (velocity_wide <<< 1) : 36'sd0) +
        (lead_up[2] ? (velocity_wide <<< 2) : 36'sd0) +
        (lead_up[3] ? (velocity_wide <<< 3) : 36'sd0);
    wire signed [35:0] down_advance =
        (lead_down[0] ? velocity_wide : 36'sd0) +
        (lead_down[1] ? (velocity_wide <<< 1) : 36'sd0) +
        (lead_down[2] ? (velocity_wide <<< 2) : 36'sd0) +
        (lead_down[3] ? (velocity_wide <<< 3) : 36'sd0);
    wire signed [35:0] predicted_up = rel_wide + up_advance;
    wire signed [35:0] predicted_down = rel_wide + down_advance;
    wire [31:0] pulse_limit = (last_drive == UP ? pulse_up_steps : pulse_down_steps) *
                              PULSE_STEP_CYCLES;
    // K y G deben transportar la misma ventana de ±20 cuentas; se usa para
    // la llegada estable y para B. La parada anticipada apunta al centro.
    wire at_goal = (control_rel >= goal_ext - arrival_window_ext &&
                    control_rel <= goal_ext + arrival_window_ext);
    assign relative = rel[31:0];
    assign motor_up = (drive == UP) && (mode == MANUAL ||
        (mode == READY && rel >= -bottom_window_ext && rel < max_ext));
    assign motor_down = (drive == DOWN) && (mode == MANUAL ||
        (mode == READY && rel > -bottom_window_ext && rel <= safety_max_ext));
    assign moving = motor_up || motor_down;
    assign busy = goal_active;

    always @(posedge clk) begin
        control_rel <= rel;
        position_changed_last_cycle <= (position != previous_position);
        previous_position <= position;
        if (speed_timer >= SPEED_SAMPLE_CYCLES - 1) begin
            speed_timer <= 0;
            velocity <= control_rel - sampled_rel;
            sampled_rel <= control_rel;
        end else speed_timer <= speed_timer + 1'b1;
        if (mode == READY && goal_active && drive == STOP && !settling)
            speed_timer <= 0;
        if (mode == READY && goal_active && drive != STOP) begin
            if (pulse_active || abs_error <= 33'sd40) pwm_percent <= 70;
            else if (abs_error >= 33'sd160) pwm_percent <= 75;
            else if (abs_error >= 33'sd128) pwm_percent <= 74;
            else if (abs_error >= 33'sd96) pwm_percent <= 73;
            else if (abs_error >= 33'sd64) pwm_percent <= 72;
            else if (abs_error >= 33'sd32) pwm_percent <= 71;
            else pwm_percent <= 70;
        end else pwm_percent <= 75;
        if (drive != STOP || position_changed_last_cycle) stable_timer <= 0;
        else if (stable_timer < STABLE_CYCLES) stable_timer <= stable_timer + 1'b1;

        if (mode == DISARMED || mode == FAULT) watchdog <= 0;
        else if (watchdog < WATCHDOG_CYCLES) watchdog <= watchdog + 1'b1;
        if (command_valid && command == "H") watchdog <= 0;

        if (drive == STOP && dead_timer != 0)
            dead_timer <= dead_timer - 1'b1;
        if (position != previous_position) stall_timer <= 0;
        else if (drive != STOP && stall_timer < STALL_CYCLES)
            stall_timer <= stall_timer + 1'b1;

        if (mode == MANUAL && drive != STOP) begin
            if (jog_timer < WATCHDOG_CYCLES) jog_timer <= jog_timer + 1'b1;
            else begin
                drive <= STOP;
                dead_timer <= DEAD_CYCLES;
            end
        end
        if (mode == READY && goal_active) begin
            if (progress_started && progress_timer < PROGRESS_CYCLES)
                progress_timer <= progress_timer + 1'b1;
            if (drive != STOP && pulse_active && pulse_timer < pulse_limit)
                pulse_timer <= pulse_timer + 1'b1;
        end
        if (mode == READY && goal_active) begin
            if (rel < -bottom_window_ext || rel > safety_max_ext) begin
                mode <= FAULT; drive <= STOP; goal_active <= 0; hold_active <= 0;
                pulse_active <= 0;
                fault_code <= 1;
            end else if (stall_timer >= STALL_CYCLES) begin
                mode <= FAULT; drive <= STOP; goal_active <= 0; hold_active <= 0;
                pulse_active <= 0;
                fault_code <= 2;
            end else if (drive == STOP && compute_excess) begin
                settled_excess <= measured_excess;
                compute_excess <= 0;
                evaluate_settled <= 1;
            end else if (drive == STOP && evaluate_settled) begin
                evaluate_settled <= 0;
                if (settled_excess == 0) begin
                    goal_active <= 0;
                end else if (progress_started && progress_timer >= PROGRESS_CYCLES &&
                             !meaningful_progress) begin
                    // El resultado no convergente conserva la referencia y el modo R.
                    goal_active <= 0;
                    hold_active <= 0;
                    fault_code <= hold_correction ? 4'd6 : 4'd5;
                end else begin
                    if (!progress_started || meaningful_progress) begin
                        best_excess <= settled_excess;
                        progress_started <= 1;
                        progress_timer <= 0;
                    end
                    if (last_drive == UP) begin
                        if (settled_rel < goal_ext - arrival_window_ext) begin
                            if (last_pulsed && pulse_up_steps < 4'd8)
                                pulse_up_steps <= pulse_up_steps + 1'b1;
                            else if (!last_pulsed && lead_up != 0)
                                lead_up <= lead_up - 1'b1;
                        end else if (settled_rel > goal_ext + arrival_window_ext) begin
                            if (last_pulsed && pulse_up_steps > 4'd2)
                                pulse_up_steps <= pulse_up_steps - 1'b1;
                            else if (!last_pulsed && lead_up < 4'd15)
                                lead_up <= lead_up + 1'b1;
                        end
                    end else if (last_drive == DOWN) begin
                        if (settled_rel > goal_ext + arrival_window_ext) begin
                            if (last_pulsed && pulse_down_steps < 4'd8)
                                pulse_down_steps <= pulse_down_steps + 1'b1;
                            else if (!last_pulsed && lead_down != 0)
                                lead_down <= lead_down - 1'b1;
                        end else if (settled_rel < goal_ext - arrival_window_ext) begin
                            if (last_pulsed && pulse_down_steps > 4'd2)
                                pulse_down_steps <= pulse_down_steps - 1'b1;
                            else if (!last_pulsed && lead_down < 4'd15)
                                lead_down <= lead_down + 1'b1;
                        end
                    end
                end
            end else if ((drive == UP && control_rel < drive_origin - 33'sd8) ||
                         (drive == DOWN && control_rel > drive_origin + 33'sd8)) begin
                mode <= FAULT; drive <= STOP; goal_active <= 0; hold_active <= 0;
                pulse_active <= 0;
                fault_code <= 3;
            end else if (drive != STOP && pulse_active && pulse_timer >= pulse_limit) begin
                drive <= STOP;
                pulse_active <= 0;
                dead_timer <= DEAD_CYCLES;
                settling <= 1;
                stable_timer <= 0;
            end else if (drive == UP && !pulse_active && predicted_up >= goal_ext) begin
                drive <= STOP;
                dead_timer <= DEAD_CYCLES;
                settling <= 1;
                stable_timer <= 0;
            end else if (drive == DOWN && !pulse_active && predicted_down <= goal_ext) begin
                drive <= STOP;
                dead_timer <= DEAD_CYCLES;
                settling <= 1;
                stable_timer <= 0;
            end else if (drive == STOP && settling && stable_timer >= STABLE_CYCLES &&
                         !position_changed_last_cycle && position == previous_position) begin
                settled_rel <= control_rel;
                compute_excess <= 1;
                settling <= 0;
            end else if (drive == STOP && !settling && dead_timer == 0) begin
                if (at_goal) begin
                    settling <= 1;
                    stable_timer <= 0;
                end else if (control_rel < goal_ext) begin
                    drive <= UP;
                    drive_origin <= control_rel;
                    last_drive <= UP;
                    last_pulsed <= abs_error <= 33'sd40;
                    pulse_active <= abs_error <= 33'sd40;
                    pulse_timer <= 0;
                    velocity <= 0;
                    sampled_rel <= control_rel;
                end else begin
                    drive <= DOWN;
                    drive_origin <= control_rel;
                    last_drive <= DOWN;
                    last_pulsed <= abs_error <= 33'sd40;
                    pulse_active <= abs_error <= 33'sd40;
                    pulse_timer <= 0;
                    velocity <= 0;
                    sampled_rel <= control_rel;
                end
            end
        end
        if (mode == READY && hold_active && !goal_active && !at_goal) begin
            goal_active <= 1;
            settling <= 0;
            hold_correction <= 1;
            progress_started <= 0;
            progress_timer <= 0;
            stall_timer <= 0;
        end
        if (mode == READY && (rel < -bottom_window_ext || rel > safety_max_ext)) begin
            mode <= FAULT;
            drive <= STOP;
            goal_active <= 0;
            hold_active <= 0;
            pulse_active <= 0;
            compute_excess <= 0;
            evaluate_settled <= 0;
            fault_code <= 1;
        end

        if (mode != DISARMED && mode != FAULT && watchdog >= WATCHDOG_CYCLES) begin
            mode <= DISARMED;
            drive <= STOP;
            goal_active <= 0;
            hold_active <= 0;
            pulse_active <= 0;
            compute_excess <= 0;
            evaluate_settled <= 0;
            settling <= 0;
            span <= 0;
            goal <= 0;
        end

        if (command_valid) begin
            if (command == "D") begin
                mode <= DISARMED;
                fault_code <= 0;
                drive <= STOP;
                goal_active <= 0;
                hold_active <= 0;
                pulse_active <= 0;
                compute_excess <= 0;
                evaluate_settled <= 0;
                stall_timer <= 0;
                settling <= 0;
                span <= 0;
                goal <= 0;
                dead_timer <= DEAD_CYCLES;
            end else if (command == "M" && mode == DISARMED) begin
                mode <= MANUAL;
                fault_code <= 0;
                drive <= STOP;
                goal_active <= 0;
                hold_active <= 0;
                pulse_active <= 0;
                compute_excess <= 0;
                evaluate_settled <= 0;
                stall_timer <= 0;
                watchdog <= 0;
            end else if (command == "K" && (mode == DISARMED || mode == CONFIGURED) &&
                         cfg_abs_span >= 500 && cfg_abs_span <= 20000 &&
                         config_margin >= 1 && config_margin <= 500 &&
                         ({15'd0,config_margin,1'b0} < cfg_abs_span) &&
                         config_tolerance >= 1 && config_tolerance <= 50 &&
                         config_tolerance <= config_margin &&
                         config_bottom_window == config_tolerance) begin
                span <= config_span;
                margin <= config_margin;
                tolerance <= config_tolerance;
                bottom_window <= config_bottom_window;
                arrival_window <= 0;
                goal <= 0;
                goal_active <= 0;
                hold_active <= 0;
                pulse_active <= 0;
                compute_excess <= 0;
                evaluate_settled <= 0;
                stall_timer <= 0;
                drive <= STOP;
                mode <= CONFIGURED;
                fault_code <= 0;
                lead_up <= 4'd5;
                lead_down <= 4'd7;
                pulse_up_steps <= 4'd4;
                pulse_down_steps <= 4'd4;
                stable_timer <= 0;
                watchdog <= 0;
            end else if (command == "Z" && mode == CONFIGURED &&
                         drive == STOP && stable_timer >= STABLE_CYCLES) begin
                reference <= position;
                goal <= 0;
                mode <= READY;
                watchdog <= 0;
            end else if (command == "A" && mode == CONFIGURED &&
                         drive == STOP && stable_timer >= STABLE_CYCLES) begin
                reference <= absolute_bottom;
                goal <= 0;
                mode <= READY;
                watchdog <= 0;
            end else if (command == "I" && config_session_token != 0) begin
                session_token <= config_session_token;
            end else if (command == "X") begin
                drive <= STOP;
                goal_active <= 0;
                hold_active <= 0;
                pulse_active <= 0;
                compute_excess <= 0;
                evaluate_settled <= 0;
                settling <= 0;
                stall_timer <= 0;
                fault_code <= 0;
                dead_timer <= DEAD_CYCLES;
            end else if (command == "G" && mode == READY && drive == STOP &&
                         !goal_active && config_goal <= max_goal &&
                         config_goal_window == tolerance) begin
                goal <= config_goal;
                arrival_window <= config_goal_window;
                fault_code <= 0;
                goal_active <= 1;
                hold_active <= 1;
                hold_correction <= 0;
                pulse_active <= 0;
                compute_excess <= 0;
                evaluate_settled <= 0;
                settling <= 0;
                progress_started <= 0;
                progress_timer <= 0;
                stall_timer <= 0;
                watchdog <= 0;
            end else if (mode == MANUAL && (command == "W" || command == "S")) begin
                watchdog <= 0;
                if ((drive == UP && command == "W") || (drive == DOWN && command == "S")) begin
                    jog_timer <= 0;
                end else if (drive != STOP) begin
                    drive <= STOP;
                    dead_timer <= DEAD_CYCLES;
                end else if (dead_timer == 0) begin
                    drive <= (command == "W") ? UP : DOWN;
                    jog_timer <= 0;
                end
            end
        end
    end
endmodule
