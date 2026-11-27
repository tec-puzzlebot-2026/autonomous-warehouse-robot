module quadrature_decoder (
    input  wire clk,
    input  wire enc_a,
    input  wire enc_b,

    output reg signed [31:0] position,
    output reg               direction
);
    initial begin
        position = 32'sd0;
        direction = 1'b0;
    end

    // Sincronizadores: el encoder es asíncrono respecto al FPGA
    reg a_meta = 1'b0;
    reg a_sync = 1'b0;
    reg b_meta = 1'b0;
    reg b_sync = 1'b0;

    reg [1:0] prev_ab = 2'b00;
    reg initialized = 1'b0;

    wire [1:0] current_ab;

    assign current_ab = {a_sync, b_sync};

    always @(posedge clk) begin

        // Doble sincronización
        a_meta <= enc_a;
        a_sync <= a_meta;

        b_meta <= enc_b;
        b_sync <= b_meta;

        // Evita contar una transición falsa al arrancar
        if (!initialized) begin
            prev_ab <= current_ab;
            initialized <= 1'b1;
        end
        else begin

            case ({prev_ab, current_ab})

                // Sentido 1
                4'b0001,
                4'b0111,
                4'b1110,
                4'b1000: begin
                    position  <= position + 1'b1;
                    direction <= 1'b1;
                end

                // Sentido contrario
                4'b0010,
                4'b1011,
                4'b1101,
                4'b0100: begin
                    position  <= position - 1'b1;
                    direction <= 1'b0;
                end

                default: begin
                    // Sin movimiento o transición inválida
                end

            endcase

            prev_ab <= current_ab;
        end
    end

endmodule
