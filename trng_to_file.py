import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from math import log2

# Konfiguracja
L = 8
alpha = 1.99999
epsilon = 0.05
total_bytes_needed = 13 * 1024 * 1024  # 13 MB
bytes_per_block = 32  # 256-bit
blocks_needed = total_bytes_needed // bytes_per_block
samples_per_block = 8
total_samples = blocks_needed * samples_per_block
transient_samples = 10000  # Odrzucenie pierwszych próbek

def tent_map(x):
    return alpha * x if x < 0.5 else alpha * (1 - x)

def record_audio(samples=total_samples, fs=44100):
    print("Nagrywanie audio...")
    audio = sd.rec(frames=samples + transient_samples, samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio[transient_samples:].flatten()

def extract_3lsbs(samples):
    return np.bitwise_and(samples, 0b111)

def perturb(xi, r):
    return ((0.071428571 * r) + xi) * 0.666666667

def ccml_iteration(x):
    new_x = np.zeros_like(x)
    for i in range(L):
        left = x[(i - 1) % L]
        right = x[(i + 1) % L]
        new_x[i] = (1 - epsilon) * tent_map(x[i]) + (epsilon / 2) * (tent_map(left) + tent_map(right))
    return new_x

def swap_32bits(z):
    upper = (z >> 32) & 0xFFFFFFFF
    lower = z & 0xFFFFFFFF
    return (lower << 32) | upper

def calculate_entropy(byte_sequence):
    counts = np.bincount(byte_sequence, minlength=256)
    probabilities = counts / len(byte_sequence)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def generate_blocks(audio_bits, block_count):
    x = np.array([0.141592, 0.653589, 0.793238, 0.462643,
                  0.383279, 0.502884, 0.197169, 0.399375])
    all_bytes = bytearray()
    hex_lines = []

    for i in range(block_count):
        r_values = audio_bits[i * 8: (i + 1) * 8]
        for j in range(L):
            x[j] = perturb(x[j], r_values[j])

        for _ in range(4):
            x = ccml_iteration(x)

        z = [np.frombuffer(np.float64(val).tobytes(), dtype=np.uint64)[0] for val in x]
        result = []
        for k in range(L // 2):
            zi = int(z[k]) ^ swap_32bits(int(z[k + L // 2]))
            result.append(zi)

        for val in result:
            b = val.to_bytes(8, byteorder='big')
            all_bytes += b
            hex_lines.append(b.hex())

    return all_bytes, hex_lines

# Główna część programu
print("Rozpoczynanie generowania TRNG...")
try:
    audio = record_audio()
except Exception as e:
    print("Błąd mikrofonu:", e)
    exit()

audio_bits = extract_3lsbs(audio)
output_bytes, output_hex_lines = generate_blocks(audio_bits, blocks_needed)

# Obliczenie entropii
byte_values = np.frombuffer(output_bytes, dtype=np.uint8)
entropy = calculate_entropy(byte_values)
print("Entropia wyjścia:", round(entropy, 6), "bitów na bajt")

# Zapis plików
with open("trng_output.bin", "wb") as f:
    f.write(output_bytes)

with open("trng_output.txt", "w") as f:
    f.write("Etrpopia: " + str(round(entropy, 6)) + "\n")
    f.write('\n'.join(output_hex_lines))

with open("trng_output_dec.txt", "w") as f:
    f.write('\n'.join(map(str, byte_values)))

# Rysowanie histogramu
plt.figure(figsize=(10, 6))
plt.hist(byte_values, bins=256, range=(0, 255), color='skyblue', edgecolor='black')
plt.title("Histogram wartości bajtów TRNG")
plt.xlabel("Wartość bajtu (0–255)")
plt.ylabel("Liczba wystąpień")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("trng_histogram.png")

print("Zakończono generowanie TRNG")
print("Zapisane pliki: trng_output.bin, trng_output.txt, trng_output_dec.txt, trng_histogram.png")