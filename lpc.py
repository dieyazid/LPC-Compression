import numpy as np
from scipy.signal import lfilter, hamming
from scipy.linalg import toeplitz, solve_toeplitz
import heapq
from collections import Counter

def lpc_analysis(signal, frame_size, overlap, order):
    frames = frame_signal(signal, frame_size, overlap)
    lpc_coefficients = []

    for frame in frames:
        # if frame values all  = 0 pass
        if np.all(frame == 0):
            continue
        windowed_frame = frame * hamming(frame_size)
        a = autocorrelation_method(windowed_frame, order)
        lpc_coefficients.append(a)

    return np.array(lpc_coefficients)

def autocorrelation_method(frame, order):
    # Compute the autocorrelation function using numpy's correlate function
    corr = np.correlate(frame, frame, mode='full')
    
    # Extract the relevant part of the autocorrelation function
    r = corr[len(frame)-1:len(frame)+order]

    # Create the first column of the Toeplitz matrix
    c = r[:-1]
    
    # Solve the Toeplitz system of equations using scipy's solve_toeplitz
    a = solve_toeplitz(c, r[1:])
    
    return a

def compute_excitation(signal, lpc_coefficients):
    excitation = np.zeros_like(signal)

    for i, a in enumerate(lpc_coefficients):
        inverse_filter = [1] + [-ai for ai in a[1:]]
        excitation[i * len(a):(i + 1) * len(a)] = lfilter(a, 1, signal[i * len(a):(i + 1) * len(a)])

    return excitation

def quantize_and_code(lpc_coefficients, excitation_signal):
    # Perform quantization and coding for LPC coefficients
    quantized_lpc = quantize_lpc(lpc_coefficients)
    # coded_lpc = code_lpc(quantized_lpc)
    coded_lpc, huffman_dict = code_lpc(quantized_lpc)

    # Perform quantization and coding for the excitation signal
    quantized_excitation = quantize_excitation(excitation_signal)
    coded_excitation = code_excitation(quantized_excitation)

    return coded_lpc, coded_excitation, huffman_dict

def quantize_lpc(lpc_coefficients, num_levels=256):
    # Find the minimum and maximum values of LPC coefficients
    min_val = np.min(lpc_coefficients)
    max_val = np.max(lpc_coefficients)

    # Calculate the quantization step
    step_size = (max_val - min_val) / (num_levels - 1)

    # Quantize the LPC coefficients
    quantized_lpc = np.round((lpc_coefficients - min_val) / step_size) * step_size + min_val

    return quantized_lpc


def huffman_encode(symbol_freq):
    heap = [[weight, [sym, ""]] for sym, weight in symbol_freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huffman_dict = dict(heap[0][1:])
    return huffman_dict

def code_lpc(quantized_lpc):
    # Convert the NumPy array to a list of tuples
    lpc_freq = Counter(map(tuple, quantized_lpc))


    # Create a Huffman tree and dictionary for encoding
    huffman_dict = huffman_encode(lpc_freq)

    # Encode the quantized LPC coefficients using Huffman coding
    coded_lpc = ''.join(huffman_dict[tuple(coeff)] for coeff in quantized_lpc)


    return coded_lpc, huffman_dict

def frame_signal(signal, frame_size, overlap):
    frames = []
    hop_size = frame_size - overlap
    for i in range(0, len(signal) - frame_size + 1, hop_size):
        frames.append(signal[i:i + frame_size])
    return frames

def quantize_excitation(excitation_signal):
    # Perform quantization of excitation signal
    # Replace this with your quantization algorithm
    # This might involve scalar quantization or another approach
    quantized_excitation = excitation_signal

    return quantized_excitation

def code_excitation(quantized_excitation):
    # Convert the NumPy array to a list of tuples
    excitation_freq = Counter(quantized_excitation)

    # Create a Huffman tree and dictionary for encoding
    huffman_dict = huffman_encode(excitation_freq)

    # Encode the quantized excitation using Huffman coding
    coded_excitation = ''.join(huffman_dict[sample] for sample in quantized_excitation)

    return coded_excitation
