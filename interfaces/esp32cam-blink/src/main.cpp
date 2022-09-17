#include <Arduino.h>

// Set LED_BUILTIN if it is not defined by Arduino framework
#define LED_BUILTIN 4

void setup()
{
  Serial.begin(9600);
  Serial.setDebugOutput(true);
  Serial.println("Setup starting...");
  Serial.flush();

  pinMode(LED_BUILTIN, OUTPUT);
}

void loop()
{
  digitalWrite(LED_BUILTIN, HIGH);
  delay(5);
  digitalWrite(LED_BUILTIN, LOW);
  
  Serial.println("Waiting 5s...");
  Serial.flush();
  delay(5000);
}
