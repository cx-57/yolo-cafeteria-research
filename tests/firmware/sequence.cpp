#include "Servo.h"
#include "../../firmware/waste_sorter/waste_sorter.ino"
void tick(unsigned long ms) { for (unsigned long i=0; i<ms; ++i) { ++fakeNow; loop(); } }
void send(const std::string &s) { Serial.inject(s); tick(5); }
int main() {
  setup();
  assert(phase == IDLE);
  assert(Serial.output == "READY 60 90 120\n");
  assert(writes.size() == 2 && writes[0].angle == 90 && writes[1].angle == 90);
  writes.clear(); Serial.output.clear();
  send("NOPE\nSORT 0\nSORT 4\nSORT 1 extra\n\n");
  assert(phase == IDLE && writes.empty());
  assert(Serial.output.find("ERROR UNKNOWN") != std::string::npos);
  send(std::string(60, 'x') + "SORT 1\n");
  assert(phase == IDLE && writes.empty());
  assert(Serial.output.find("ERROR MALFORMED") != std::string::npos);
  send("SORT "); tick(1001); send("1\n");
  assert(phase == IDLE && writes.empty());
  assert(Serial.output.find("ERROR LINE_TIMEOUT") != std::string::npos);
  for(int bin=1; bin<=3; ++bin) {
    writes.clear(); Serial.output.clear();
    unsigned long start=fakeNow;
    send("SORT " + std::to_string(bin) + "\r\n");
    assert(phase != IDLE);
    send("SORT 2\nPING\nCENTER\n");
    assert(Serial.output.find("ERROR BUSY") != std::string::npos);
    tick(10000);
    assert(phase == IDLE && chuteAngle == 90 && trapdoorAngle == 90);
    assert(Serial.output.find("DONE SORT " + std::to_string(bin)) != std::string::npos);
    unsigned long aimAt=start, openAt=0, closedAt=0, firstReturn=0;
    int lastChute=90, lastTrap=90, openCount=0;
    unsigned long lastTrapAt=0, lastChuteAt=0;
    for (auto e : writes) {
      if(e.pin == 10) {
        assert(abs(e.angle-lastChute)==1);
        if(lastChuteAt) assert(e.at-lastChuteAt>=25);
        lastChuteAt=e.at;
        if(!openAt) aimAt=e.at;
        else if(!firstReturn) firstReturn=e.at;
        lastChute=e.angle;
      } else if(e.pin == 9) {
        if(e.angle==160) {
          assert(lastChute==CHUTE_ANGLES[bin-1]);
          assert(e.at-aimAt>=1000);
          assert(lastTrap==90);
          openAt=e.at; ++openCount;
        } else {
          assert(openAt && e.angle==lastTrap-1);
          assert(e.at-lastTrapAt>=25);
          if(lastTrap==160) assert(e.at-openAt>=1500);
          if(e.angle==90) closedAt=e.at;
        }
        lastTrapAt=e.at; lastTrap=e.angle;
      } else assert(false);
    }
    assert(openCount==1 && closedAt>openAt && lastTrap==90 && lastChute==90);
    if(bin!=2) assert(firstReturn-closedAt>=1000);
  }
  Serial.output.clear(); send("CENTER\n");
  assert(Serial.output == "BUSY CENTER\nDONE CENTER\n");
  Serial.output.clear(); send("PING\n");
  assert(Serial.output == "READY 60 90 120\n");
  // millis wraparound must not break elapsed-time comparisons.
  fakeNow=0xFFFFFF00UL;
  send("SORT 1\n"); tick(10000);
  assert(phase==IDLE && trapdoorAngle==90 && chuteAngle==90);
  return 0;
}
