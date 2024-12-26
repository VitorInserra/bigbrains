import usb.core
import usb.util
import time

dev = usb.core.find(idVendor=0x1234, idProduct=0xED02)

if dev:
    print(dev)
else:
    print("Device not found")

if dev.is_kernel_driver_active(1):
    dev.detach_kernel_driver(1)

dev.set_configuration()

cfg = dev.get_active_configuration()
intf = cfg[(1, 0)]
endpoint = intf[0]

try:
    start = time.time()
    i = 0
    while i < 256:
        data = dev.read(endpoint.bEndpointAddress, endpoint.wMaxPacketSize, timeout=100)
        print("Data received:", data)
        i += 1
    print(time.time() - start)
except usb.core.USBTimeoutError:
    print("Timeout: No data received.")
