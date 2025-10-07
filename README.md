# Advanced Drone Audio Enhancer

A Python-based system for enhancing speech audio by suppressing both **drone noise** (characterized by specific tonal/harmonic patterns) and **white noise**. The system is designed to be trained on the DREGON dataset (or similar structured audio files) for effective noise modeling.

##  Underlying Principles (How it Works)

The system operates based on the principles of **Spectral Noise Reduction**, combining two powerful techniques: **Noise Modeling** and **Advanced Filtering**.

### 1. Noise Modeling and Training
The core idea is to *characterize* the noise and the clean speech.
* **Spectral Feature Extraction:** The code analyzes the audio by converting it into the frequency domain (using **STFT** - Short-Time Fourier Transform) and computes the **average power spectrum** for clean speech, drone noise, and white noise.
* **Energy Statistics:** The system learns the average energy and spectral shape of the noise components and the clean speech from the training data (e.g., dedicated noise-only files and clean speech files).

### 2. Advanced Filtering
Once the noise is characterized, spectral filters are created to selectively attenuate the noise components while preserving the speech.

* **Adaptive Spectral Subtraction:** This technique estimates the noise spectrum and *subtracts* it from the noisy signal's spectrum on a frame-by-frame basis. The filter applied is **adaptive**, meaning the degree of suppression is adjusted based on the estimated **Signal-to-Noise Ratio (SNR)** at each frequency band. Frequencies with strong drone or white noise relative to speech get heavier suppression.
* **Wiener Filtering:** This is a statistically optimal filtering method that minimizes the mean square error between the estimated clean signal and the true clean signal. The Wiener gain is calculated using the ratio of the estimated **clean speech power spectrum** to the **total noisy speech power spectrum**, providing a smooth and perceptually sound noise reduction.
* **Combined and Post-Processing:** The two filtering methods are applied sequentially for a robust enhancement. Finally, the audio is **post-processed** (normalized and band-pass filtered) to ensure a high-quality output.

### Prerequisites
You need to have the necessary Python libraries installed and access to the **DREGON dataset** (or your own structured audio files) for training.

1.  **Install Libraries:**
    ```bash
    pip install numpy librosa soundfile scipy
    ```
2.  **Audio Data:**
    Place the training audio files (e.g., from the DREGON dataset) in the same directory as the Python script. The script expects the following files for robust training:
    * `2min_TIMIT.wav` (or similar clean speech)
    * `DREGON_free-flight_nosource_room1.wav` (or similar pure drone noise)
    * `2min_white_noise.wav` (or similar white noise)
    * `DREGON_free-flight_speech-high_room1.wav` and `DREGON_free-flight_speech-low_room1.wav` (for further noise and pre-drone speech segments)

---
### Operating


```bash
python drone_enhancer.py
```

P.S. One can upload whatever files they want and change the final line of the code to the name of the file. 

