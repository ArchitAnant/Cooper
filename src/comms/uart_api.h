#ifndef UART_API_H
#define UART_API_H

#include <stddef.h>
#include "../dcnn/includes/model_runner.h"

#define UART_HANDSHAKE 0xAA

int uart_init();
int uart_wait_for_host(void);
int uart_fill_rx_buf(float *buf, size_t count);
int uart_send_result(PredResult result);

#endif