#include "stm32f10x.h"
#include "Encoder.h"
#include "Motor.h"
#include "PWM.h"
#include "OLED.h"
#include "Serial.h"
#include "Delay.h"
#include "LED.h"

// 定义标志位
#define INIT_FLAG        116    // 初始化标志位
#define START_FLAG       110    // 开始标志位
#define RETURN_FLAG      101    // 返回标志位

// 定义小车控制参数
#define FORWARD_SPEED    100     // 前进速度
#define TURN_SPEED       80     // 转弯速度
#define FORWARD_PULSES   1800    // 前进2米所需的编码器脉冲数（根据实际测试调整）
#define TURN_TIME        1000     // 左转90度所需的时间（毫秒）

// 全局变量
uint8_t receivedFlag = 0;

// 函数声明
void Car_Init(void);
void Car_Forward(uint16_t pulses);
void Car_TurnLeft(void);
void Car_TurnBack(void);
void Car_WaitForFlag(uint8_t expectedFlag);
void LED_Shine(void);

int main(void) {
    // 初始化
    Car_Init();

    // 显示初始化信息
    OLED_ShowString(1, 1, "Car Ready");

    while (1) {
		// 等待初始化标志位
        Car_WaitForFlag(INIT_FLAG);
		
		LED_Shine();
		
        // 等待开始标志位
        Car_WaitForFlag(START_FLAG);
		
		LED_Shine();

        // 前进2米
		OLED_Clear();
        OLED_ShowString(1, 1, "Forward");
        Car_Forward(FORWARD_PULSES);

        // 左转90度
		OLED_Clear();
        OLED_ShowString(1, 1, "Left");
        Car_TurnLeft();

		LED_Shine();
		
        // 发送图像采集标志位
		OLED_Clear();
        OLED_ShowString(1, 1, "Capture");
        Serial_SendString("arrived");

        // 等待返回标志位
        Car_WaitForFlag(RETURN_FLAG);
		
		LED_Shine();

        // 左转90度
		OLED_Clear();
        OLED_ShowString(1, 1, "Left Again");
        Car_TurnLeft();

        // 返回2米
		OLED_Clear();
        OLED_ShowString(1, 1, "Returning");
        Car_Forward(FORWARD_PULSES);
		
		// 调头
		OLED_Clear();
        OLED_ShowString(1, 1, "back");
        Car_TurnBack();
		
		LED_Shine();

        // 显示完成信息
		OLED_Clear();
        OLED_ShowString(1, 1, "Complete");
    }
}

void Car_Init(void) {
    // 初始化各模块
    Encoder_Init(TIM3, GPIOA, GPIO_Pin_6, GPIO_Pin_7); // 根据实际接线调整参数
    Motor_Init();
    PWM_Init();
    Serial_Init();
    OLED_Init();
	LED_Init();

    // 清屏
    OLED_Clear();
}

void Car_Forward(uint16_t pulses) {
    // 设置电机方向为前进
    Motor_SetPWM(FORWARD_SPEED, FORWARD_SPEED, FORWARD_SPEED, FORWARD_SPEED);

	OLED_ShowNum(3,1,Encoder_Get(TIM3),4);
    // 等待编码器脉冲达到目标值
    Delay_s(10);

    // 停止电机
    Motor_SetPWM(0, 0, 0, 0);
}

void Car_TurnLeft(void) {
    // 设置电机方向为左转（左轮反转，右轮正转）
    Motor_SetPWM(TURN_SPEED, -TURN_SPEED, TURN_SPEED, -TURN_SPEED);

    // 延时等待转弯完成
    //Delay_ms(TURN_TIME);
    Delay_s(10);

    // 停止电机
    Motor_SetPWM(0, 0, 0, 0);
}

void Car_TurnBack(void) {
    // 设置电机方向为右转（右轮反转，左轮正转）
    Motor_SetPWM(-TURN_SPEED, TURN_SPEED, -TURN_SPEED, TURN_SPEED);

    // 延时等待转弯完成
    //Delay_ms(2*TURN_TIME);
    Delay_s(10);

    // 停止电机
    Motor_SetPWM(0, 0, 0, 0);
}

void Car_WaitForFlag(uint8_t expectedFlag) {
    while (1) {
        // 检查是否收到标志位
        receivedFlag=Serial_GetRxData();
		OLED_ShowNum(2,1,receivedFlag,4);
        if (receivedFlag==expectedFlag) {
			break;
        }
    }
}

void LED_Shine(void){
	LED_ON();
	Delay_ms(500);
	LED_OFF();
}
