import numpy as np
import matplotlib.pyplot as plt

G = np.array([[1, 0, 0, 0, 1, 1, 0],
              [0, 1, 0, 0, 1, 0, 1],
              [0, 0, 1, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1]])

def encode_and_modulate(data, G):
    encoded = np.mod(np.dot(data, G), 2)
    # BPSK Modulation: Map 0 -> +1, 1 -> -1
    modulated = 1 - 2 * encoded
    return modulated

def add_noise(codewords, ebn0, rate):
    snr_linear = 10 ** (ebn0 / 10) * rate
    noise_std = np.sqrt(1 / (2 * snr_linear))
    noise = noise_std * np.random.randn(*codewords.shape)
    return codewords + noise, noise_std

def mld(received, G, noise_std):
    # Generate the codebook
    codebook = np.array([1 - 2 * np.mod(np.dot([i >> 3, (i >> 2) & 1, (i >> 1) & 1, i & 1], G), 2) for i in range(16)])
    decoded = []
    for r in received:
        # Calculate the likelihood for each codeword (considering BPSK symbols)
        likelihoods = np.exp(-np.sum((r - codebook) ** 2, axis=1) / (2 * noise_std ** 2))
        # Select the codeword with the highest likelihood
        decoded.append(codebook[np.argmax(likelihoods)])
    return np.array(decoded)

def calculate_bler(original, decoded):
    # Map back from BPSK to binary for comparison
    decoded_binary = (1 - decoded) // 2
    # Compare only the first 4 bits (information bits) of the original and decoded codewords
    return np.mean(np.any(original != decoded_binary[:, :4], axis=1))

def simulate_hamming_bler(ebn0_values, rate, num_blocks=500000):
    bler_values = []
    for ebn0 in ebn0_values:
        data = np.random.randint(0, 2, (num_blocks, 4))
        encoded_data = encode_and_modulate(data, G)
        noisy_data, noise_std = add_noise(encoded_data, ebn0, rate)
        decoded_data = mld(noisy_data, G, noise_std)
        bler = calculate_bler(data, decoded_data)
        bler_values.append(bler)
    return bler_values

ebn0_values = np.arange(1, 8, .2)
rate = 4 / 7  # Code rate
bler_values = simulate_hamming_bler(ebn0_values, rate)

# Plot the BLER vs Eb/N0
plt.figure(figsize=(10, 6))
plt.semilogy(ebn0_values, bler_values, marker='o')
plt.xlabel('$E_b/N_0$ (dB)')
plt.ylabel('BLER')
plt.title('BLER vs $E_b/N_0$ for Hamming (7,4) Code with MLD')
plt.grid(True)
plt.show()