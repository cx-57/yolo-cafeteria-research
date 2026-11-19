/* Uno R3: trapdoor signal D9, chute signal D10. Servos use regulated
   external 5V; supply ground, servo grounds and Uno ground are common.
   9600 baud ASCII newline protocol. No sorting cycle runs at reset.
   Keep CHUTE_ANGLES in sync with configs/sorter.json; PING verifies them.
   servo.write sets a target, not measured position. Calibrate unloaded. */
#include <Servo.h>
#include <string.h>

const byte TRAPDOOR_PIN = 9;
const byte CHUTE_PIN = 10;
const int TRAPDOOR_CLOSED = 90;
const int TRAPDOOR_OPEN = 160;
const int CHUTE_CENTER = 90;
const int CHUTE_ANGLES[3] = {60, 90, 120}; // physical slots 1/2/3, NOT categories
const unsigned long STEP_MS = 25;
const unsigned long CHUTE_SETTLE_MS = 1000;
const unsigned long TRAPDOOR_HOLD_MS = 1500;
// write(160) is immediate; allow actual opening travel before the hold.
const unsigned long TRAPDOOR_OPEN_TRAVEL_MS = 300;
const unsigned long CLOSED_SETTLE_MS = 1000;
const unsigned long SERIAL_LINE_TIMEOUT_MS = 1000;

Servo trapdoor, chute;
enum Phase { IDLE, AIMING, SETTLING, OPEN_HOLD, CLOSING, CLOSED_WAIT, RETURNING, CENTERING };
Phase phase = IDLE;
int chuteAngle = CHUTE_CENTER;
int trapdoorAngle = TRAPDOOR_CLOSED;
byte selectedBin = 0;
unsigned long phaseAt = 0, stepAt = 0, lastByteAt = 0;
char input[24];
byte used = 0;
bool discardLine = false;

void ready() {
  Serial.print(F("READY"));
  for (byte i = 0; i < 3; ++i) { Serial.print(' '); Serial.print(CHUTE_ANGLES[i]); }
  Serial.println();
}

bool moveStep(Servo &servo, int &angle, int target, unsigned long now) {
  if (angle == target) return true;
  if (now - stepAt >= STEP_MS) {
    angle += (angle < target ? 1 : -1);
    servo.write(angle);
    stepAt = now;
  }
  return angle == target;
}

void handleCommand(unsigned long now) {
  if (!strcmp(input, "PING")) {
    if (phase == IDLE) ready();
    else Serial.println(F("ERROR BUSY"));
    return;
  }
  if (phase != IDLE) { Serial.println(F("ERROR BUSY")); return; }
  if (!strcmp(input, "CENTER")) {
    phase = CENTERING;
    stepAt = now;
    Serial.println(F("BUSY CENTER"));
    return;
  }
  if (strlen(input) == 6 && !strncmp(input, "SORT ", 5) && input[5] >= '1' && input[5] <= '3') {
    selectedBin = input[5] - '0';
    phase = AIMING;
    stepAt = now;
    Serial.print(F("BUSY SORT ")); Serial.println(selectedBin);
    return;
  }
  Serial.println(F("ERROR UNKNOWN"));
}

void readCommands(unsigned long now) {
  if (used && now - lastByteAt >= SERIAL_LINE_TIMEOUT_MS) {
    used = 0;
    discardLine = true; // do not execute a late suffix as a fresh command
    Serial.println(F("ERROR LINE_TIMEOUT"));
  }
  // Bound work per loop so malformed traffic cannot starve the servo sequence.
  for (byte budget = 0; budget < 32 && Serial.available(); ++budget) {
    char c = Serial.read();
    lastByteAt = now;
    if (c == '\n') {
      if (!discardLine && used) {
        input[used] = '\0';
        handleCommand(now);
      } else if (!discardLine) Serial.println(F("ERROR EMPTY"));
      used = 0;
      discardLine = false;
    } else if (c == '\r') {
      // Accept CRLF from Serial Monitor as well as LF from Python.
    } else if (!discardLine) {
      if (c < 32 || c > 126 || used >= sizeof(input)-1) {
        used = 0;
        discardLine = true;
        Serial.println(F("ERROR MALFORMED"));
      } else input[used++] = c;
    }
  }
}

void updateCycle(unsigned long now) {
  switch (phase) {
    case IDLE: break;
    case AIMING:
      if (moveStep(chute, chuteAngle, CHUTE_ANGLES[selectedBin-1], now)) {
        phase = SETTLING; phaseAt = now;
      }
      break;
    case SETTLING:
      if (now - phaseAt >= CHUTE_SETTLE_MS) {
        trapdoorAngle = TRAPDOOR_OPEN;
        trapdoor.write(TRAPDOOR_OPEN); // direct command, no incremental opening
        phase = OPEN_HOLD; phaseAt = now;
      }
      break;
    case OPEN_HOLD:
      if (now - phaseAt >= TRAPDOOR_OPEN_TRAVEL_MS + TRAPDOOR_HOLD_MS) {
        phase = CLOSING; stepAt = now;
      }
      break;
    case CLOSING:
      if (moveStep(trapdoor, trapdoorAngle, TRAPDOOR_CLOSED, now)) {
        phase = CLOSED_WAIT; phaseAt = now;
      }
      break;
    case CLOSED_WAIT:
      if (now - phaseAt >= CLOSED_SETTLE_MS) { phase = RETURNING; stepAt = now; }
      break;
    case RETURNING:
      if (moveStep(chute, chuteAngle, CHUTE_CENTER, now)) {
        phase = IDLE;
        Serial.print(F("DONE SORT ")); Serial.println(selectedBin);
      }
      break;
    case CENTERING:
      if (moveStep(chute, chuteAngle, CHUTE_CENTER, now)) {
        phase = IDLE; Serial.println(F("DONE CENTER"));
      }
      break;
  }
}

void setup() {
  // Set target values before attaching so startup pulses use the safe targets.
  trapdoor.write(TRAPDOOR_CLOSED);
  chute.write(CHUTE_CENTER);
  trapdoor.attach(TRAPDOOR_PIN);
  chute.attach(CHUTE_PIN);
  Serial.begin(9600);
  ready();
}

void loop() {
  unsigned long now = millis();
  readCommands(now);
  updateCycle(now);
}
