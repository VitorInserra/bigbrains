import usb.core
import usb.util
import time
import numpy as np


def decode_packet(packet):
    """
    Decode a single packet of EEG data.
    Assumes:
    - 1 header byte
    - 14 channels × 2 bytes (16-bit integers)
    """
    header = int(packet[0], 16)  # Convert header from hex to integer
    raw_channel_data = packet[1:29]  # First 28 bytes after the header

    # Decode 14 channels (16-bit values)
    channels = []
    for i in range(0, len(raw_channel_data), 2):
        # Combine two bytes into a 16-bit integer
        channel_value = (int(raw_channel_data[i], 16) << 8) | int(
            raw_channel_data[i + 1], 16
        )
        channels.append(channel_value)

    return header, channels


dev = usb.core.find(idVendor=, idProduct=)

if dev.is_kernel_driver_active(1):
    dev.detach_kernel_driver(1)

dev.set_configuration()

cfg = dev.get_active_configuration()
intf = cfg[(1, 0)]
endpoint = intf[0]

ls = []

start = time.time()

for i in range(60):
    j = 0
    while j < 256:
        ls.append(
            [
                hex(b)
                for b in dev.read(
                    endpoint.bEndpointAddress, endpoint.wMaxPacketSize, timeout=100
                )
            ]
        )
        j += 1
print(time.time() - start)

decoded_data = []

for packet in ls:
    try:
        header, channels = decode_packet(packet)
        decoded_data.append({"header": header, "channels": channels})
    except Exception as e:
        print(f"Error decoding packet: {e}")
        continue



import matplotlib.pyplot as plt

# Initialize 14 empty lists for each channel
channels_data = [[] for _ in range(14)]

# Populate channel data
for packet in decoded_data:
    for i, value in enumerate(packet["channels"]):
        channels_data[i].append(value)

# Time axis
time = [i / 256 for i in range(len(channels_data[0]))]  # Time in seconds at 256 SPS
scale_factor = 0.1275  # Example: Convert raw values to µV
scaled_channels = [[value * scale_factor for value in channel] for channel in channels_data]

# Plot each channel
for i, channel in enumerate(scaled_channels):
    channel_data = scaled_channels[0]  # Replace with your desired channel
sampling_rate = 256  # Sampling rate in Hz (256 SPS)

# Perform FFT
fft_result = np.fft.fft(channel_data)
fft_magnitude = np.abs(fft_result)  # Magnitude of FFT
fft_power = fft_magnitude ** 2  # Power spectrum

# Frequency axis
freqs = np.fft.fftfreq(len(channel_data), d=1/sampling_rate)

# Only keep positive frequencies (real-valued signals)
positive_freqs = freqs[:len(freqs)//2]
positive_power = fft_power[:len(freqs)//2]

# Plot the frequency spectrum
plt.figure(figsize=(12, 8))
plt.plot(positive_freqs, positive_power, label="Power Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Frequency Spectrum of EEG Signal (Channel 1)")
plt.legend()
plt.show()

