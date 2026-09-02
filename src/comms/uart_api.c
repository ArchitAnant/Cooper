#include "uart_api.h"

#include <errno.h>
#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/uart.h>

#define UART_NODE DT_CHOSEN(zephyr_console)

static const struct device *uart_dev;

int uart_init()
{
    uart_dev = DEVICE_DT_GET(UART_NODE);

    if(!device_is_ready(uart_dev))
        return -ENODEV;

    return 0;
}

int uart_wait_for_host(void)
{
    uint8_t byte;

    while (1) {
        if (uart_poll_in(uart_dev, &byte) == 0) {
            if (byte == UART_HANDSHAKE) {
                uart_poll_out(uart_dev, UART_HANDSHAKE);
                return 0;
            }
        }

        k_msleep(10);
    }
}

int uart_fill_rx_buf(float *buf, size_t count)
{
    if(buf == NULL)
        return -EINVAL;

    uint8_t *data = (uint8_t *)buf;
    size_t len = count * sizeof(float);

    for(size_t i = 0; i < len; i++){
        while (uart_poll_in(uart_dev, &data[i]) < 0);
    }

    return 0;
}

int uart_send_result(PredResult result)
{
    int32_t class_id = result.class_id;
    float confidence = result.confidence;

    const uint8_t *class_data = (const uint8_t *)&class_id;
    const uint8_t *conf_data = (const uint8_t *)&confidence;

    for (size_t i = 0; i < sizeof(class_id); i++)
        uart_poll_out(uart_dev, class_data[i]);

    for (size_t i = 0; i < sizeof(confidence); i++)
        uart_poll_out(uart_dev, conf_data[i]);

    return 0;
}