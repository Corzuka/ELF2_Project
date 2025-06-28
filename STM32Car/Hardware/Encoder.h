#ifndef __ENCODER_H
#define __ENCODER_H

void Encoder_Init(TIM_TypeDef* TIMx, GPIO_TypeDef* GPIOx, uint16_t CH_1, uint16_t CH_2);
int16_t Encoder_Get(TIM_TypeDef* TIMx);

#endif
