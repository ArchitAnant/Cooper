#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include "dcnn/model_runner.h"

int main(){
    int ret;

    printk("[+] Boot Up\n");
    ret = init_runtime();
    if(ret){
        printk("[-] Error init model\n");
        return 0;
    }

    float temp_array[490] = {0};
    ret = run_inference(temp_array);
    if(ret!=-1)
        printk("[+] successful\n");
    else
        printk("[-] error inference");

    printk("Exiting...");
    return 0;
}