#include <Servo.h>

Servo bottomServo;
int target = 0;
int currentPosition = 0;

void setup() {
  Serial.begin(9600);
  bottomServo.attach(9);
  bottomServo.write(target);
}

bool checkValid(int pos) {
  if(pos >= 0 && pos <= 180) {
    return true;
  } else {
    return false;
  }
}

void Print(int pos) {
  Serial.print("target = ");
  Serial.print(pos);
  Serial.print(", current = ");
  Serial.print(currentPosition);
}

void loop() {
  if(Serial.available()) {
    target = Serial.readString().toInt();
    if(checkValid(target)) {
      Print(target);
      int pos = currentPosition;
      if(target >= currentPosition) {
        Serial.write("target >= currentPosition");
        while(pos < target) {
          pos++;
          bottomServo.write(pos);
          Serial.println(pos);
          delay(10);
        }
      } else {
        Serial.write("target < currentPosition");
        while(target < pos) {
          pos--;
          bottomServo.write(pos);
          Serial.println(pos);
          delay(10);
        }
      }
      currentPosition = target;
    }
  }
}
