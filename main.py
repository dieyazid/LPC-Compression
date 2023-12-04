from scipy.io import wavfile
from lpc import quantize_and_code, compute_excitation, lpc_analysis

# Read the original audio file
fs, original_signal = wavfile.read('3.wav')

# Assuming you want to use a frame size of 20 ms, overlap of 10 ms, and an LPC order of 12
frame_size = int(0.02 * fs)  # 20 ms
overlap = int(0.01 * fs)  # 10 ms
lpc_order = 12

# Call the LPC analysis function
lpc_coefficients = lpc_analysis(original_signal, frame_size, overlap, lpc_order)

# Compute the excitation signal
excitation_signal = compute_excitation(original_signal, lpc_coefficients)

# Quantize and code the LPC coefficients and excitation signal
coded_lpc, coded_excitation, huffman_dict = quantize_and_code(lpc_coefficients, excitation_signal)

# Write compressed data to a file
with open('compressed_data.bin', 'wb') as file:
    file.write(coded_lpc.encode('utf-8'))  # Convert to bytes if needed
    file.write(coded_excitation.encode('utf-8'))
    # Also, save the Huffman dictionary for decoding if necessary
    # You might need to serialize it to a format like JSON before writing to the file

# Calculate the size of the original and compressed audio files
original_size = original_signal.nbytes
compressed_size = len(coded_lpc.encode('utf-8')) + len(coded_excitation.encode('utf-8'))  # Adjust this based on the actual encoding

# Print the size comparison
print(f"Original Size: {original_size} bytes")
print(f"Compressed Size: {compressed_size} bytes")
print(f"Compression Ratio: {original_size / compressed_size:.2f}")
