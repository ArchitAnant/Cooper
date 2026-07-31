# Cooper

### Building
---
```bash
west build -p always -b rpi_pico2/rp2350a/m33 -S cdc-acm-console .
```

### Console/Debugging
---
```bash
ls /dev/cu.usbmodem*
```
if you see a device like `/dev/cu.usbmodem101`, you can connect to it using `minicom`:
```bash
minicom -D /dev/cu.usbmodem101 -b 115200
```