char SerialData;
int pin = 13;

void setup() {
  pinMode(pin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if(Serial.available() > 0) {
    SerialData = Serial.read();
    Serial.println(SerialData);
  }
  if(SerialData == '1') {
    digitalWrite(pin, HIGH);
  }
  else if(SerialData == '0')  {
    digitalWrite(pin, LOW);
  }
}
