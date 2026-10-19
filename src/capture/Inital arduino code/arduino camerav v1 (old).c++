/*
  Microscope ring light — automated segment control
  --------------------------------------------------------------
  Fully automated version: no manual keyboard input needed.
  Send a segment number as ASCII text followed by a newline
  (e.g. "0\n", "1\n", ... "5\n") over serial (9600 baud) and the
  Arduino immediately lights that segment and replies "OK <n>".
  This is meant to be driven by a Python script (e.g. pyserial)
  that syncs LED segment changes with camera triggers.

  >>> amountOfQuadrants below MUST match the amountOfQuadrants
      variable in the Python script — they are two separate
      programs and this value is not shared automatically. <

  Wiring:
    Strip Red  (VCC) -> 5V supply + (both the 4-wire bundle AND the
                         black/red power-injection pair)
    Strip White(GND) -> 5V supply - AND Arduino GND (both bundles)
    Strip Black(GND) -> 5V supply - (power-injection pair)
    Strip Green(DAT) -> Arduino pin 6, through a 200 ohm resistor
    Strip Blue (BIN/BOUT) -> not connected (fault-tolerance line, unused)

  Library needed: FastLED (Arduino IDE > Tools > Manage Libraries > "FastLED")
*/

#include <FastLED.h>

#define LED_PIN        6
#define NUM_LEDS       44

// Set this to the same number of segments as amountOfQuadrants in Python
const int amountOfQuadrants = 4;   // e.g. 4, 6, 8, ...

const int LEDS_PER_QUAD = NUM_LEDS / amountOfQuadrants;
#define BRIGHTNESS     255
#define RING_COLOR     CRGB::White

CRGB leds[NUM_LEDS];
int quadrant = 0;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(50);

  FastLED.addLeds<WS2812B, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(BRIGHTNESS);

  showQuadrant(quadrant);

  // Let Python know we're alive and which quadrant is lit.
  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    // Read a full line, e.g. "3\n"
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.length() == 0) {
      return;
    }

    if (line == "x") {
      // Turn everything off, useful between runs
      FastLED.clear();
      FastLED.show();
      Serial.println("OK OFF");
      return;
    }

    if (line == "n") {
      // Manual/relative advance, still supported if you want it
      quadrant = (quadrant + 1) % amountOfQuadrants;
      showQuadrant(quadrant);
      Serial.print("OK ");
      Serial.println(quadrant);
      return;
    }

    if (line == "f") {
      // Turn on all LEDs
      fill_solid(leds, NUM_LEDS, RING_COLOR);
      FastLED.show();
      Serial.println("OK ALL");
      return;
    }

    if (line.startsWith("p")) {
      // 2X2 mode: one quadrant + the opposite one, e.g. "p1"
      int requested = line.substring(1).toInt();
      if (requested >= 0 && requested < amountOfQuadrants) {
        quadrant = requested;
        showPair(quadrant);
        Serial.print("OK PAIR ");
        Serial.print(quadrant);
        Serial.print("+");
        Serial.println((quadrant + amountOfQuadrants / 2) % amountOfQuadrants);
      } else {
        Serial.print("ERR out of range: ");
        Serial.println(line);
      }
      return;
    }

    if (line == "f") {
      // Turn on all LEDs
      fill_solid(leds, NUM_LEDS, RING_COLOR);
      FastLED.show();
      Serial.println("OK ALL");
      return;
    }

    // Otherwise treat it as a direct segment number
    int requested = line.toInt();
    if (requested >= 0 && requested < amountOfQuadrants) {
      quadrant = requested;
      showQuadrant(quadrant);
      Serial.print("OK ");
      Serial.println(quadrant);
    } else {
      Serial.print("ERR out of range: ");
      Serial.println(line);
    }
  }
}

void showQuadrant(int q) {
  FastLED.clear();

  int startLed = q * LEDS_PER_QUAD;
  int endLed = startLed + LEDS_PER_QUAD;

  for (int i = startLed; i < endLed; i++) {
    leds[i] = RING_COLOR;
  }

  FastLED.show();
}

void showPair(int q) {
  FastLED.clear();

  int opposite = (q + amountOfQuadrants / 2) % amountOfQuadrants;
  int quads[2] = { q, opposite };

  for (int k = 0; k < 2; k++) {
    int startLed = quads[k] * LEDS_PER_QUAD;
    int endLed = startLed + LEDS_PER_QUAD;
    for (int i = startLed; i < endLed; i++) {
      leds[i] = RING_COLOR;
    }
  }

  FastLED.show();
}