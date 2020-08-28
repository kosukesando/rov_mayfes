#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>
#include <ESP32Servo.h>
#include "config.h"

//ESP32の注意点として、書き込み時にBootボタンを押さないとコンパイルエラーが出ることがある
//
//参考
//Example集: https://examples.blynk.cc/?board=ESP32&shield=ESP32%20WiFi&example=GettingStarted%2FVirtualPinWrite
//ESP32のPWM: https://randomnerdtutorials.com/esp32-pwm-arduino-ide/
//ESP32のPWM: https://www.mgo-tec.com/blog-entry-ledc-pwm-arduino-esp32.html#title02

//トークン発行して入れてください
char auth[] = BLINK_API_KEY;

// Your WiFi credentials.
// Set password to "" for open networks.
char ssid[] = WIFI_SSID;
char pass[] =  WIFI_PASSWORD;

Servo servo1;
const int maxUs = 1900;
const int minUs = 1100;
const int servo1Pin = 26;
const int servo1Period = 50;

const int led_pin = 14;    // GPIOのピン番号
const int freq = 5000;     //PWMの周波数
const int led_channel = 0; // ESP32はPWMチャンネルが任意のピンで16チャンネルまで設定できる
const int resolution = 8;  //

int servo1Us = 1500;

//アプリ側でVirtual Pinに書き込みがあるたびに呼ばれる関数
//paramがV6に書き込まれたデータで、asInt()でInt型として処理 asFloatとかも色々ある
//BLYNK_WRITE(V6)
//{
//  int dutycycle = param.asInt();
//  ledcWrite(led_channel, dutycycle);
//}

BLYNK_WRITE(V8)
{
  servo1Us = param.asInt();
}

//BlynkTimer timer1;
BlynkTimer timer2;

void myTimerEvent()
{
  Blynk.virtualWrite(V5, millis() / 1000);
}

int count = 0;

void servoLoop()
{
  servo1.writeMicroseconds(servo1Us);
}

void setup()
{
  delay(100);
  // Debug console
  Serial.begin(9600);
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  servo1.setPeriodHertz(servo1Period);
  servo1.attach(servo1Pin, minUs, maxUs);
  servo1.writeMicroseconds(1500);
  delay(5000);
  servo1.writeMicroseconds(1500);
  Blynk.begin(auth, ssid, pass);

  //  ledcSetup(led_channel, freq, resolution);
  //  ledcAttachPin(led_pin, led_channel);
  //
  //  // Setup a function to be called every second
  //  timer1.setInterval(1000L, myTimerEvent);
  timer2.setInterval(20L, servoLoop);
}

void loop()
{
  Blynk.run();
  //  timer1.run(); // Initiates BlynkTimer
  timer2.run();
}
