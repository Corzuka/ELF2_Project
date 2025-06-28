#include "stm32f10x.h"
#include "PWM.h"

void Motor_Init(void) {
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA | RCC_APB2Periph_GPIOB | RCC_APB2Periph_GPIOC, ENABLE);
    
    GPIO_InitTypeDef GPIO_InitStructure;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_4 | GPIO_Pin_5;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
	
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_4 | GPIO_Pin_5;
    GPIO_Init(GPIOC, &GPIO_InitStructure);
    
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0 | GPIO_Pin_1 | GPIO_Pin_12 | GPIO_Pin_13;
    GPIO_Init(GPIOB, &GPIO_InitStructure);
}

void Motor_SetPWM(int8_t PWM1, int8_t PWM2, int8_t PWM3, int8_t PWM4) {
    if (PWM1 >= 0) {
        GPIO_SetBits(GPIOA, GPIO_Pin_5);
        GPIO_ResetBits(GPIOA, GPIO_Pin_4);
        PWM_SetCompare1(PWM1);
    } else {
        GPIO_ResetBits(GPIOA, GPIO_Pin_5);
        GPIO_SetBits(GPIOA, GPIO_Pin_4);
        PWM_SetCompare1(-PWM1);
    }
    
    if (PWM2 >= 0) {
        GPIO_SetBits(GPIOC, GPIO_Pin_5);
        GPIO_ResetBits(GPIOC, GPIO_Pin_4);
        PWM_SetCompare2(PWM2);
    } else {
        GPIO_ResetBits(GPIOC, GPIO_Pin_5);
        GPIO_SetBits(GPIOC, GPIO_Pin_4);
        PWM_SetCompare2(-PWM2);
    }
    
	if (PWM3 >= 0) {
        GPIO_SetBits(GPIOB, GPIO_Pin_0);
        GPIO_ResetBits(GPIOB, GPIO_Pin_1);
        PWM_SetCompare3(PWM3);
    } else {
        GPIO_ResetBits(GPIOB, GPIO_Pin_0);
        GPIO_SetBits(GPIOB, GPIO_Pin_1);
        PWM_SetCompare3(-PWM3);
    }
    
    if (PWM4 >= 0) {
        GPIO_SetBits(GPIOB, GPIO_Pin_13);
        GPIO_ResetBits(GPIOB, GPIO_Pin_12);
        PWM_SetCompare4(PWM4);
    } else {
        GPIO_ResetBits(GPIOB, GPIO_Pin_13);
        GPIO_SetBits(GPIOB, GPIO_Pin_12);
        PWM_SetCompare4(-PWM4);
    }
}
