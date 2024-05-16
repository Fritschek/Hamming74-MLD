# Hamming Code Simulation

This repository contains a Python script that simulates the performance of the Hamming (7,4) code under various Eb/N0 conditions using Maximum Likelihood Decoding (MLD). The script calculates and plots the Block Error Rate (BLER) as a function of Eb/N0.

## Overview

Hamming codes are a class of error-correcting codes that can detect and correct single-bit errors. This script performs the following steps:
1. Encodes random binary data using the Hamming (7,4) code.
2. Modulates the encoded data using Binary Phase Shift Keying (BPSK).
3. Adds white Gaussian noise to the modulated signal.
4. Decodes the noisy signal using Maximum Likelihood Decoding.
5. Calculates the BLER and plots it against Eb/N0.

## Prerequisites

- Python 3.x
- `numpy`
- `matplotlib`
