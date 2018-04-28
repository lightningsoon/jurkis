#include <LobotServoController.h>
#define CTRL_SER Serial1 //控制器串口
#define COM_SER Serial //电脑串口
LobotServoController myse(CTRL_SER);
const int input_params_num=7;
String comdata = "";
int init_numdata[input_params_num] = {1468,600,1300,1000,1550,500,1000};//初始位置
int numdata[input_params_num] = {1468,600,1300,1000,1550,500,1000};//位置
bool mark = true;
//numdata是分拆之后的数字数组
int temp;
LobotServo servos[6];
int speed=2000;

int j = 0;
void setup() {

  pinMode(13, OUTPUT);
  COM_SER.begin(115200);
  CTRL_SER.begin(9600);
  COM_SER.flush();
//  delay(1000);
//  COM_SER.println("Controller connected");
  COM_SER.println("Successfully initialized");
/*
  //  myse.runActionGroup(100,0);  //不断运行100号动作组
  //  delay(5000);
  //  myse.stopActionGroup(); //停止动作组运行
  //  delay(2000);
  //  myse.setActionGroupSpeed(100,200); //设置100号动作组运行速度为200%
  //  delay(2000);
  //  myse.runActionGroup(100,5);  //运行100号动作组 5次
  //  myse.moveServo(0,1500,1000); //0号舵机1000ms移动至1500位置
  //  delay(2000);
  
  //1468,600,1200,800,1500,1060
*/
  setServos(init_numdata);
  delay(200);
  myse.moveServos(servos, 6, 2000);
  //初始化
  //控制6个舵机，移动时间1000ms，1号舵机至1468位置，2号舵机至500位置，3号舵机至1500位置，
  //4号舵机至1500位置，5号舵机至1500位置,6号舵机至1060位置

/*
    LobotServo servos[2];   //舵机ID位置结构数组
    servos[0].ID = 2;       //2号舵机
    servos[0].Position = 1400;  //1400位置
    servos[1].ID = 4;       //4号舵机
    servos[1].Position = 700;  //700位置
    myse.moveServos(servos,2,1000);  //控制两个舵机，移动时间1000ms,ID和位置有servos指定
*/
zero();
}

void loop() {
  //初始化
  
  COM_SER.flush();
  while (!COM_SER.available())
  delay(300);
  comdata=COM_SER.readStringUntil('!');
//  COM_SER.println(comdata);
  if(comdata=="homing")
  {
    setServos(init_numdata);
    myse.moveServos(servos, 6, 2000);
  }
  else
  {
    for (int i = 0,j=0; i < comdata.length() ; i++)
    {
      temp=comdata[i] - '0';
      if ((temp>=0) && (temp<=9))
      {
        numdata[j] = numdata[j] * 10 + temp;
      }
      else
      {
//        COM_SER.println(numdata[j]);
        j++;
      }
    }
    
    speed=numdata[6];
//    COM_SER.println(speed);
    setServos(numdata);
    myse.moveServos(servos, 6, speed); //控制6个舵机，移动时间2000ms,ID和位置有servos指定
    delay(speed);
    COM_SER.println('1');
  }
  zero();
}
void zero()
{
    //归零准备下次操作
  for (int i = 0; i < input_params_num; i++)
  {
    numdata[i] = 0;
  }
  comdata = String("");//清空
  temp=0;
  speed = 800;
}
void setServos(int numdata[])
{
  for (int k = 0; k < 6; k++)
  {
    servos[k].ID = k + 1;
    servos[k].Position = numdata[k];
  }
}

