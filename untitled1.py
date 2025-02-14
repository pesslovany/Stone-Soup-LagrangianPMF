#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:06:08 2025

@author: matoujak
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# Define 2D grid
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
dx = x[1] - x[0]  # Grid step size
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Define two 2D Gaussian PDFs
sigma1, sigma2 = 1.0, 1.5
pdf1 = np.exp(-0.5 * ((X / sigma1) ** 2 + (Y / sigma1) ** 2)) / (2 * np.pi * sigma1**2)
pdf1 /= np.sum(pdf1) * dx * dy  # Normalize

pdf2 = np.exp(-0.5 * ((X / sigma2) ** 2 + (Y / sigma2) ** 2)) / (2 * np.pi * sigma2**2)
pdf2 /= np.sum(pdf2) * dx * dy  # Normalize

pdf1 *= dx * dy  # Scale for proper convolution

# Perform 2D convolution using fftconvolve
conv_result = fftconvolve(pdf1, pdf2, mode="same")

# Check normalization
print("Normalization check:", np.sum(conv_result) * dx * dy)  # Should be close to 1

# Plot results
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(pdf1, extent=[-10, 10, -10, 10], origin="lower", cmap="viridis")
ax[0].set_title("PDF 1 (σ=1)")
ax[1].imshow(pdf2, extent=[-10, 10, -10, 10], origin="lower", cmap="viridis")
ax[1].set_title("PDF 2 (σ=1.5)")
ax[2].imshow(conv_result, extent=[-10, 10, -10, 10], origin="lower", cmap="viridis")
ax[2].set_title("Convolution Result")
plt.show()
