// Nano Every: dos Sharp IR independientes en A0 y A1.
// USB Serial: CSV de diagnostico a 115200. Serial1 D1/TX: 9600 8N1
// hacia el divisor de 5 V a 3.3 V y el pin 30 de la Tang.
// Trama A9: q1 adc1_lo adc1_hi spread1_lo spread1_hi
//           q2 adc2_lo adc2_hi spread2_lo spread2_hi XOR (12 bytes).
// q=0: ventana estable; q=1: calentamiento o lectura inestable.

const byte INICIO_TRAMA = 0xA9;
const byte MUESTRAS = 9;
const unsigned long PERIODO_MUESTRA_MS = 60;
const unsigned long PERIODO_TRAMA_MS = 300;
const uint16_t DISPERSION_MAX_ADC = 20;

struct CanalIR {
  byte pin;
  uint16_t ventana[MUESTRAS];
  byte cantidad;
  byte siguienteIndice;
};
CanalIR canales[2] = {{A0, {}, 0, 0}, {A1, {}, 0, 0}};
unsigned long ultimaMuestraMs = 0;
unsigned long ultimaTramaMs = 0;

void filtrar(const CanalIR &canal, uint16_t &mediana,
             uint16_t &dispersion, bool &estable) {
  uint16_t ordenadas[MUESTRAS];
  for (byte i = 0; i < canal.cantidad; i++) ordenadas[i] = canal.ventana[i];
  for (byte i = 1; i < canal.cantidad; i++) {
    uint16_t actual = ordenadas[i];
    int j = i - 1;
    while (j >= 0 && ordenadas[j] > actual) {
      ordenadas[j + 1] = ordenadas[j];
      --j;
    }
    ordenadas[j + 1] = actual;
  }
  mediana = ordenadas[canal.cantidad / 2];
  if (canal.cantidad < MUESTRAS) {
    dispersion = 0xFFFF;
    estable = false;
  } else {
    dispersion = ordenadas[MUESTRAS - 2] - ordenadas[1];
    estable = dispersion <= DISPERSION_MAX_ADC;
  }
}

void enviarTrama(const uint16_t adc[2], const uint16_t dispersion[2],
                 const bool estable[2]) {
  byte trama[12] = {
      INICIO_TRAMA,
      byte(estable[0] ? 0 : 1), byte(adc[0]), byte(adc[0] >> 8),
      byte(dispersion[0]), byte(dispersion[0] >> 8),
      byte(estable[1] ? 0 : 1), byte(adc[1]), byte(adc[1] >> 8),
      byte(dispersion[1]), byte(dispersion[1] >> 8), 0};
  for (byte i = 0; i < 11; i++) trama[11] ^= trama[i];
  Serial1.write(trama, sizeof(trama));
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(9600);
  Serial.println(F("tiempo_ms,adc_ir1,dispersion_ir1,calidad_ir1,adc_ir2,dispersion_ir2,calidad_ir2"));
  ultimaMuestraMs = millis() - PERIODO_MUESTRA_MS;
}

void loop() {
  unsigned long ahora = millis();
  if (ahora - ultimaMuestraMs >= PERIODO_MUESTRA_MS) {
    ultimaMuestraMs = ahora;
    for (byte i = 0; i < 2; i++) {
      CanalIR &canal = canales[i];
      // Descartar la primera conversion tras cambiar el multiplexor ADC.
      analogRead(canal.pin);
      canal.ventana[canal.siguienteIndice] = analogRead(canal.pin);
      canal.siguienteIndice = (canal.siguienteIndice + 1) % MUESTRAS;
      if (canal.cantidad < MUESTRAS) ++canal.cantidad;
    }
  }
  if (canales[0].cantidad == 0 || ahora - ultimaTramaMs < PERIODO_TRAMA_MS) return;
  ultimaTramaMs = ahora;

  uint16_t adc[2], dispersion[2];
  bool estable[2];
  for (byte i = 0; i < 2; i++)
    filtrar(canales[i], adc[i], dispersion[i], estable[i]);
  enviarTrama(adc, dispersion, estable);
  Serial.print(ahora);
  for (byte i = 0; i < 2; i++) {
    Serial.print(','); Serial.print(adc[i]);
    Serial.print(','); Serial.print(dispersion[i]);
    Serial.print(','); Serial.print(estable[i] ? F("ESTABLE") : F("INESTABLE"));
  }
  Serial.println();
}
