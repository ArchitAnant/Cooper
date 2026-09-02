#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "comms/uart_api.h"
#include "processing/includes/clip.h"
#include "processing/includes/mfcc.h"
#include "dcnn/includes/model_runner.h"

#define BUF_LEN 16000

static float mfcc_features[NUM_MFCC * 49];
float rx_buf[BUF_LEN];

int main(void) {
    int ret;
    uart_init();

    // printk("[+] Boot Up Completed\n");
    // printk("[+] Holding till connection.\n");

    
    // printk("[+] Connection Established.\n");
    // printk("[+] Init Execu runtime\n");
    
    if (init_runtime()){
        // printk("[-] Failed to initialize Execu runtime\n");
        return 0;
    }
    
    mfcc_extractor_handle_t mfcc_handle = mfcc_create();
    // printk("[+] Entring Inference loop\n");
    
    while(1){
        uart_wait_for_host();
        ret = uart_fill_rx_buf(rx_buf, BUF_LEN);

        if(ret < 0){
            // printk("[-] UART receive failed: %d\n", ret);
            continue;
        }

        // printk("[+] Data ready proceeding with prediction\n");

        clip_audio(rx_buf);
        mfcc_process_clip(mfcc_handle, rx_buf, mfcc_features);

        PredResult res = run_inference(mfcc_features);

        // printk("[+] Prediction Result\nclass: %d\npred : %f\n",res.class_id, res.confidence);
        uart_send_result(res);

        k_sleep(K_MSEC(10));
    }
    mfcc_destroy(mfcc_handle);
    // printk("[+] Exiting...\n");
    return 0;
}