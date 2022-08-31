#include <Servo.h>
Servo joints[8];
//int joint_pin_ids[8];
const byte servoPins[] = {2, 3, 4, 5, 6, 7, 8, 9};
void setup() {
  // put your setup code here, to run once:
  int i = 0;
  for(i= 0; i<8; i++){
    joints[i].attach(servoPins[i]);
    }
}

void reset(){
  int j = 0;
  for(j = 0; j < 8; j++){
    joints[j].write(90);
    //delay(10);
    }
  }

void test(){
  int pos = 0;
  int j = 0;
  for(pos = 45; pos < 135; pos++){
    for(j = 0; j < 8; j ++){
      joints[j].write(pos);
      //delay(10);
      }
    }
  for(pos = 135; pos < 45; pos--){
    for(j = 0; j < 8; j ++){
      joints[j].write(pos);
      //delay(10);
      }
    }

    
  }
 

void loop() {
  // put your main code here, to run repeatedly:
  reset();
  test();
  reset();
}
