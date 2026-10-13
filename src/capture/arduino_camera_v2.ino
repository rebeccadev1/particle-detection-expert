/*
  Microscope ring light — uniform N/S/E/W
  --------------------------------------------------------------
  The ring is four sides (N, S, E, W). Uniform mode lights all four.
  NSEW mode lights one side: 0=N, 1=E, 2=S, 3=W, clockwise from the
  side that faces the top of the image. Driven over serial at 9600
  baud by camera_automation_v2.py. Every command is one ASCII line
  ending in '\n'. A line that is not a known command is rejected.

  Commands:
    ?     -> QUAD 4
    f     -> OK ALL     all four sides on
    0..3  -> OK <n>      that side only
    x     -> OK OFF      all LEDs off

  On boot every LED is on and the sketch sends "READY ALL".
  Re-upload this sketch after changing it.

  The strip is packages of 3 LEDs, so a side must be a multiple of 3
  or it slices a package. 44 is not a multiple of 12, so four equal
  sides cannot do that. Side mode uses the 42 LEDs that make 14 whole
  packages. Opposites match: N and S are 3 packages (9 LEDs), E and W
  are 4 packages (12 LEDs). LEDs 42 and 43 are the strip joint and stay
  off in side mode. Uniform mode still lights all 44.

  Wiring:
    Strip Red  (VCC) -> 5V supply + (both the 4-wire bundle AND the
                         black/red power-injection pair)
    Strip White(GND) -> 5V supply - AND Arduino GND (both bundles)
    Strip Black(GND) -> 5V supply - (power-injection pair)
    Strip Green(DAT) -> Arduino pin 6, through a 200 ohm resistor
    Strip Blue (BIN/BOUT) -> not connected (fault-tolerance line, unused)

  Full-white on all 44 LEDs is about 2.6 A. Use the external 5V supply.
  USB cannot power the ring.

  Library: FastLED (Arduino IDE > Tools > Manage Libraries > "FastLED")
*/

#include <FastLED.h>

#define LED_PIN 6

constexpr int NUM_LEDS = 44;
constexpr int LEDS_PER_PACKAGE = 3;
constexpr int amountOfQuadrants = 4;

// Whole packages only. N, E, S, W. Opposites are the same length.
constexpr int SIDE_START[amountOfQuadrants] = {0, 9, 21, 30};
constexpr int SIDE_COUNT[amountOfQuadrants] = {9, 12, 9, 12};

static_assert(amountOfQuadrants == 4, "ring is N, S, E, W");
static_assert(SIDE_COUNT[0] % LEDS_PER_PACKAGE == 0, "N must be whole packages");
static_assert(SIDE_COUNT[1] % LEDS_PER_PACKAGE == 0, "E must be whole packages");
static_assert(SIDE_COUNT[2] % LEDS_PER_PACKAGE == 0, "S must be whole packages");
static_assert(SIDE_COUNT[3] % LEDS_PER_PACKAGE == 0, "W must be whole packages");
static_assert(SIDE_START[0] == 0, "N starts at the first LED");
static_assert(SIDE_START[1] == SIDE_START[0] + SIDE_COUNT[0], "E follows N");
static_assert(SIDE_START[2] == SIDE_START[1] + SIDE_COUNT[1], "S follows E");
static_assert(SIDE_START[3] == SIDE_START[2] + SIDE_COUNT[2], "W follows S");
static_assert(SIDE_START[3] + SIDE_COUNT[3] == 42,
              "side mode covers the 14 whole packages");
static_assert(SIDE_COUNT[0] == SIDE_COUNT[2] && SIDE_COUNT[1] == SIDE_COUNT[3],
              "opposite sides match");

constexpr int MAX_LINE = 16;

#define BRIGHTNESS 255
#define RING_COLOR CRGB::White

CRGB leds[NUM_LEDS];

// These strips keep the new frame on the data line until the line stays
// low for longer than FastLED's built-in gap. Without this pause, the
// frame sent for N only becomes visible when E is sent, and "all on"
// only appears on the second click.
void pushLeds() {
  FastLED.show();
  delay(2);
}

void showAll() {
  fill_solid(leds, NUM_LEDS, RING_COLOR);
  pushLeds();
}

void showQuadrant(int q) {
  FastLED.clear();
  if (q < 0 || q >= amountOfQuadrants) {
    pushLeds();
    return;
  }
  int startLed = SIDE_START[q];
  int count = SIDE_COUNT[q];
  for (int i = startLed; i < startLed + count; i++) {
    leds[i] = RING_COLOR;
  }
  pushLeds();
}

bool parseSide(const String& text, int& out) {
  if (text.length() != 1 || text.charAt(0) < '0' || text.charAt(0) > '9') {
    return false;
  }
  out = text.charAt(0) - '0';
  return out < amountOfQuadrants;
}

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(50);

  FastLED.addLeds<WS2812B, LED_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(BRIGHTNESS);

  // Say READY before the LEDs. A full-white frame can sag the supply and
  // reset the board before this line, and then the host never sees it.
  Serial.println("READY ALL");
  showAll();
}

void loop() {
  if (Serial.available() <= 0) {
    return;
  }

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) {
    return;
  }
  if (line.length() > MAX_LINE) {
    Serial.println("ERR line too long");
    return;
  }

  if (line == "?") {
    Serial.print("QUAD ");
    Serial.println(amountOfQuadrants);
    return;
  }

  if (line == "x") {
    FastLED.clear();
    pushLeds();
    Serial.println("OK OFF");
    return;
  }

  if (line == "f") {
    showAll();
    Serial.println("OK ALL");
    return;
  }

  int side = 0;
  if (parseSide(line, side)) {
    showQuadrant(side);
    Serial.print("OK ");
    Serial.println(side);
    return;
  }

  Serial.print("ERR unknown command: ");
  Serial.println(line);
}
