#pragma once
#include <cassert>
#include <cstdint>
#include <deque>
#include <sstream>
#include <string>
#include <vector>
using byte = uint8_t;
#define F(x) x
inline unsigned long fakeNow = 0;
inline unsigned long millis() { return fakeNow; }
struct Event { unsigned long at; int pin; int angle; };
inline std::vector<Event> writes;
class Servo {
  int pin = -1;
public:
  void attach(int value) { pin = value; }
  void write(int angle) { writes.push_back({fakeNow, pin, angle}); }
};
struct FakeSerial {
  std::deque<char> input;
  std::string output;
  void begin(int baud) { assert(baud == 9600); }
  int available() { return int(input.size()); }
  char read() { char c=input.front(); input.pop_front(); return c; }
  void print(const char *s) { output += s; }
  void print(char c) { output += c; }
  void print(byte n) { output += std::to_string(int(n)); }
  void print(int n) { output += std::to_string(n); }
  template<typename T> void println(T v) { print(v); println(); }
  void println() { output += '\n'; }
  void inject(const std::string &s) { for(char c : s) input.push_back(c); }
};
inline FakeSerial Serial;
