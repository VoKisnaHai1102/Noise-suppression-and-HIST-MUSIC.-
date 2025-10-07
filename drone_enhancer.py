import numpy as np
import librosa
import soundfile as sf
import os
from typing import Tuple, Optional, List
import warnings
from scipy import signal
from scipy.ndimage import median_filter

warnings.filterwarnings('ignore')


class AdvancedDroneAudioEnhancer:

    def __init__(self,
                 frame_length: int = 1024,
                 hop_length: int = 256,
                 sr: int = 16000,
                 n_mels: int = 80):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.sr = sr
        self.n_mels = n_mels

        self.drone_noise_spectrum = None
        self.white_noise_spectrum = None
        self.clean_speech_spectrum = None

        self.drone_suppression_filter = None
        self.white_noise_filter = None
        self.speech_enhancement_filter = None

        self.speech_energy_stats = None
        self.noise_energy_stats = None

        self.trained = False

    def compute_spectral_features(self, audio: np.ndarray) -> dict:
        stft_data = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
        magnitude = np.abs(stft_data)
        power = magnitude ** 2

        features = {
            'mean_spectrum': np.mean(power, axis=1),
            'std_spectrum': np.std(power, axis=1),
            'median_spectrum': np.median(power, axis=1),
            'max_spectrum': np.max(power, axis=1),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(S=magnitude, sr=self.sr)),
            'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(S=magnitude, sr=self.sr)),
            'spectral_flatness': np.mean(librosa.feature.spectral_flatness(S=magnitude)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio)),
            'mfccs': np.mean(librosa.feature.mfcc(S=librosa.power_to_db(magnitude),
                                                  sr=self.sr, n_mfcc=13), axis=1),
            'energy': np.mean(power)
        }

        return features, stft_data

    def analyze_noise_characteristics(self, drone_audio: np.ndarray, white_noise_audio: np.ndarray) -> dict:
        drone_features, drone_stft = self.compute_spectral_features(drone_audio)
        white_features, white_stft = self.compute_spectral_features(white_noise_audio)

        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.frame_length)

        drone_spectrum = drone_features['mean_spectrum']
        white_spectrum = white_features['mean_spectrum']

        drone_peaks, _ = signal.find_peaks(drone_spectrum, height=np.mean(drone_spectrum) * 2)
        drone_freq_peaks = freqs[drone_peaks]

        return {
            'drone_spectrum': drone_spectrum,
            'white_spectrum': white_spectrum,
            'drone_peaks': drone_freq_peaks,
            'drone_features': drone_features,
            'white_features': white_features
        }

    def extract_noise_segments(self, file_path: str, segment_type: str = 'post_drone') -> np.ndarray:
        audio, orig_sr = librosa.load(file_path, sr=self.sr)
        total_duration = len(audio) / self.sr

        if segment_type == 'pre_drone':
            if total_duration > 10:
                segment = audio[:int(10 * self.sr)]
            else:
                segment = audio[:len(audio) // 3]

        elif segment_type == 'post_drone':
            if total_duration > 15:
                segments = []
                if total_duration > 20:
                    segments.append(audio[int(10 * self.sr):int(20 * self.sr)])
                if total_duration > 40:
                    segments.append(audio[int(20 * self.sr):int(40 * self.sr)])
                if total_duration > 60:
                    segments.append(audio[int(-30 * self.sr):])

                if segments:
                    segment = np.concatenate(segments)
                else:
                    segment = audio[int(10 * self.sr):]
            else:
                segment = audio[len(audio) // 2:]

        elif segment_type == 'full':
            segment = audio

        return segment

    def create_advanced_filters(self, noise_analysis: dict):
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.frame_length)
        n_freqs = len(freqs)

        self.drone_suppression_filter = np.ones(n_freqs)
        self.white_noise_filter = np.ones(n_freqs)
        self.speech_enhancement_filter = np.ones(n_freqs)

        drone_spectrum = noise_analysis['drone_spectrum']
        white_spectrum = noise_analysis['white_spectrum']

        speech_to_drone_ratio = self.clean_speech_spectrum / (drone_spectrum + 1e-10)
        speech_to_white_ratio = self.clean_speech_spectrum / (white_spectrum + 1e-10)

        for i, freq in enumerate(freqs):
            drone_snr = speech_to_drone_ratio[i]
            white_snr = speech_to_white_ratio[i]

            if freq < 50:
                self.drone_suppression_filter[i] = 0.1
            elif freq < 500:
                if drone_snr < 1:
                    self.drone_suppression_filter[i] = 0.15
                elif drone_snr < 3:
                    self.drone_suppression_filter[i] = 0.4
                else:
                    self.drone_suppression_filter[i] = 0.7
            elif freq < 2000:
                if drone_snr < 2:
                    self.drone_suppression_filter[i] = 0.3
                else:
                    self.drone_suppression_filter[i] = 0.8
            else:
                self.drone_suppression_filter[i] = 0.9

            if white_snr < 1:
                self.white_noise_filter[i] = 0.4
            elif white_snr < 3:
                self.white_noise_filter[i] = 0.6
            else:
                self.white_noise_filter[i] = 0.85

            if 300 <= freq <= 3000:
                self.speech_enhancement_filter[i] = 1.2
            elif 100 <= freq <= 300:
                self.speech_enhancement_filter[i] = 1.1
            else:
                self.speech_enhancement_filter[i] = 1.0

        self.drone_suppression_filter = median_filter(self.drone_suppression_filter, size=3)
        self.white_noise_filter = median_filter(self.white_noise_filter, size=3)

    def train_on_dregon_dataset(self, data_folder: str = '.'):
        files = {
            'clean_speech': '2min_TIMIT.wav',
            'speech_with_drone_high': 'DREGON_free-flight_speech-high_room1.wav',
            'speech_with_drone_low': 'DREGON_free-flight_speech-low_room1.wav',
            'pure_drone': 'DREGON_free-flight_nosource_room1.wav',
            'white_noise_high': 'DREGON_free-flight_whitenoise-high_room1.wav',
            'white_noise_low': 'DREGON_free-flight_whitenoise-low_room1.wav',
            'additional_white_noise': '2min_white_noise.wav'
        }

        missing_files = []
        for name, filename in files.items():
            if not os.path.exists(os.path.join(data_folder, filename)):
                missing_files.append(filename)

        if missing_files:
            print(f"Warning: Missing files: {missing_files}")

        clean_segments = []

        if os.path.exists(os.path.join(data_folder, files['clean_speech'])):
            clean_path = os.path.join(data_folder, files['clean_speech'])
            clean_audio = self.extract_noise_segments(clean_path, 'full')
            clean_segments.append(clean_audio)

        for speech_type in ['speech_with_drone_high', 'speech_with_drone_low']:
            if files[speech_type] in [f for f in os.listdir(data_folder) if f.endswith('.wav')]:
                speech_path = os.path.join(data_folder, files[speech_type])
                pre_drone_speech = self.extract_noise_segments(speech_path, 'pre_drone')
                clean_segments.append(pre_drone_speech)

        if not clean_segments:
            raise ValueError("No clean speech data found!")

        all_clean_speech = np.concatenate(clean_segments)
        speech_features, _ = self.compute_spectral_features(all_clean_speech)
        self.clean_speech_spectrum = speech_features['mean_spectrum']
        self.speech_energy_stats = speech_features

        drone_segments = []

        if os.path.exists(os.path.join(data_folder, files['pure_drone'])):
            pure_drone_path = os.path.join(data_folder, files['pure_drone'])
            drone_segment = self.extract_noise_segments(pure_drone_path, 'post_drone')
            drone_segments.append(drone_segment)

        for speech_type in ['speech_with_drone_high', 'speech_with_drone_low']:
            if files[speech_type] in [f for f in os.listdir(data_folder) if f.endswith('.wav')]:
                speech_path = os.path.join(data_folder, files[speech_type])
                drone_segment = self.extract_noise_segments(speech_path, 'post_drone')
                drone_segments.append(drone_segment)

        if not drone_segments:
            raise ValueError("No drone noise data found!")

        combined_drone = np.concatenate(drone_segments)

        white_noise_segments = []

        for white_type in ['white_noise_high', 'white_noise_low', 'additional_white_noise']:
            if files[white_type] in [f for f in os.listdir(data_folder) if f.endswith('.wav')]:
                white_path = os.path.join(data_folder, files[white_type])
                white_segment = self.extract_noise_segments(white_path, 'full')
                white_noise_segments.append(white_segment)

        if not white_noise_segments:
            synthetic_white = np.random.normal(0, 0.1, len(combined_drone))
            white_noise_segments = [synthetic_white]

        combined_white_noise = np.concatenate(white_noise_segments)

        noise_analysis = self.analyze_noise_characteristics(combined_drone, combined_white_noise)

        self.drone_noise_spectrum = noise_analysis['drone_spectrum']
        self.white_noise_spectrum = noise_analysis['white_spectrum']

        self.create_advanced_filters(noise_analysis)

        self.validate_training()

        self.trained = True
        return True

    def validate_training(self):
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.frame_length)

        strong_drone_suppression = freqs[self.drone_suppression_filter < 0.3]
        strong_white_suppression = freqs[self.white_noise_filter < 0.5]

        speech_freqs = (freqs >= 300) & (freqs <= 3000)
        speech_suppression = self.drone_suppression_filter[speech_freqs]
        over_suppressed = np.sum(speech_suppression < 0.5)

        if over_suppressed > len(speech_suppression) * 0.3:
            print(f"Warning: Potential over-suppression in speech frequencies")

    def adaptive_spectral_subtraction(self, mixed_stft: np.ndarray) -> np.ndarray:
        magnitude = np.abs(mixed_stft)
        phase = np.angle(mixed_stft)
        n_freqs, n_frames = magnitude.shape

        enhanced_magnitude = magnitude.copy()

        drone_noise_floor = np.sqrt(self.drone_noise_spectrum)
        white_noise_floor = np.sqrt(self.white_noise_spectrum)

        for f in range(n_freqs):
            for t in range(n_frames):
                current_mag = magnitude[f, t]

                drone_noise_est = drone_noise_floor[f]
                white_noise_est = white_noise_floor[f]
                total_noise_est = np.sqrt(drone_noise_est ** 2 + white_noise_est ** 2)

                local_snr = current_mag / (total_noise_est + 1e-10)

                if local_snr > 3:
                    drone_suppression = self.drone_suppression_filter[f] * 0.8
                    white_suppression = self.white_noise_filter[f] * 0.8

                    enhanced_mag = current_mag - (1 - drone_suppression) * drone_noise_est
                    enhanced_mag = enhanced_mag - (1 - white_suppression) * white_noise_est

                    enhanced_mag = np.maximum(enhanced_mag, 0.3 * current_mag)

                elif local_snr > 1.5:
                    combined_suppression = (self.drone_suppression_filter[f] + self.white_noise_filter[f]) / 2
                    enhanced_mag = current_mag * combined_suppression

                else:
                    enhanced_mag = current_mag * self.drone_suppression_filter[f] * self.white_noise_filter[f] * 0.5

                if 300 <= librosa.fft_frequencies(sr=self.sr, n_fft=self.frame_length)[f] <= 3000:
                    enhanced_mag *= self.speech_enhancement_filter[f]

                enhanced_magnitude[f, t] = enhanced_mag

        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        return enhanced_stft

    def wiener_filter_enhancement(self, mixed_stft: np.ndarray) -> np.ndarray:
        magnitude = np.abs(mixed_stft)
        phase = np.angle(mixed_stft)

        signal_power = self.clean_speech_spectrum.reshape(-1, 1)
        drone_noise_power = self.drone_noise_spectrum.reshape(-1, 1)
        white_noise_power = self.white_noise_spectrum.reshape(-1, 1)

        total_noise_power = drone_noise_power + white_noise_power

        wiener_gain = signal_power / (signal_power + total_noise_power + 1e-10)

        enhanced_magnitude = magnitude * wiener_gain

        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        return enhanced_stft

    def enhance_audio(self, mixed_audio: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise ValueError("Model not trained! Run train_on_dregon_dataset() first.")

        mixed_stft = librosa.stft(mixed_audio, n_fft=self.frame_length, hop_length=self.hop_length)

        enhanced_stft_1 = self.adaptive_spectral_subtraction(mixed_stft)
        enhanced_stft_2 = self.wiener_filter_enhancement(enhanced_stft_1)

        enhanced_audio = librosa.istft(enhanced_stft_2, hop_length=self.hop_length)

        enhanced_audio = self.post_process_audio(enhanced_audio)

        return enhanced_audio

    def post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        sos = signal.butter(4, 80, btype='high', fs=self.sr, output='sos')
        audio = signal.sosfilt(sos, audio)

        sos = signal.butter(4, 7000, btype='low', fs=self.sr, output='sos')
        audio = signal.sosfilt(sos, audio)

        return audio

    def save_model(self, model_path: str):
        import pickle
        model_data = {
            'drone_noise_spectrum': self.drone_noise_spectrum,
            'white_noise_spectrum': self.white_noise_spectrum,
            'clean_speech_spectrum': self.clean_speech_spectrum,
            'drone_suppression_filter': self.drone_suppression_filter,
            'white_noise_filter': self.white_noise_filter,
            'speech_enhancement_filter': self.speech_enhancement_filter,
            'speech_energy_stats': self.speech_energy_stats,
            'trained': self.trained,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'sr': self.sr,
            'n_mels': self.n_mels
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, model_path: str):
        import pickle
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.drone_noise_spectrum = data['drone_noise_spectrum']
        self.white_noise_spectrum = data.get('white_noise_spectrum')
        self.clean_speech_spectrum = data['clean_speech_spectrum']
        self.drone_suppression_filter = data.get('drone_suppression_filter')
        self.white_noise_filter = data.get('white_noise_filter')
        self.speech_enhancement_filter = data.get('speech_enhancement_filter')
        self.speech_energy_stats = data.get('speech_energy_stats')
        self.trained = data['trained']
        self.frame_length = data.get('frame_length', self.frame_length)
        self.hop_length = data.get('hop_length', self.hop_length)
        self.sr = data.get('sr', self.sr)


def create_comprehensive_test(data_folder: str = '.'):
    try:
        clean_speech, _ = librosa.load(os.path.join(data_folder, '2min_TIMIT.wav'),
                                       sr=16000, duration=15.0)

        drone_audio, _ = librosa.load(os.path.join(data_folder, 'DREGON_free-flight_nosource_room1.wav'),
                                      sr=16000, offset=15.0, duration=15.0)

        white_noise_files = ['2min_white_noise.wav', 'DREGON_free-flight_whitenoise-high_room1.wav']
        white_noise = None

        for white_file in white_noise_files:
            white_path = os.path.join(data_folder, white_file)
            if os.path.exists(white_path):
                white_noise, _ = librosa.load(white_path, sr=16000, duration=15.0)
                break

        if white_noise is None:
            white_noise = np.random.normal(0, 0.05, len(clean_speech))

        min_len = min(len(clean_speech), len(drone_audio), len(white_noise))
        clean_speech = clean_speech[:min_len]
        drone_audio = drone_audio[:min_len]
        white_noise = white_noise[:min_len]

        clean_speech = clean_speech / (np.max(np.abs(clean_speech)) + 1e-10)
        drone_audio = drone_audio / (np.max(np.abs(drone_audio)) + 1e-10)
        white_noise = white_noise / (np.max(np.abs(white_noise)) + 1e-10)

        test_cases = [
            {'name': 'light_noise', 'drone_snr': 10, 'white_snr': 15},
            {'name': 'moderate_noise', 'drone_snr': 0, 'white_snr': 5},
            {'name': 'heavy_noise', 'drone_snr': -5, 'white_snr': 0}
        ]

        test_files = []

        for case in test_cases:
            drone_scale = 10 ** (-case['drone_snr'] / 20)
            white_scale = 10 ** (-case['white_snr'] / 20)

            mixed = clean_speech + drone_scale * drone_audio + white_scale * white_noise

            mixed = mixed / (np.max(np.abs(mixed)) + 1e-10) * 0.95

            output_file = f"test_{case['name']}.wav"
            sf.write(output_file, mixed, 16000)
            test_files.append(output_file)

        sf.write('test_clean_reference.wav', clean_speech, 16000)
        sf.write('test_drone_noise_only.wav', drone_audio, 16000)
        sf.write('test_white_noise_only.wav', white_noise, 16000)

        return test_files

    except Exception as e:
        print(f"Error creating test: {e}")
        return None


def train_advanced_enhancer(data_folder: str = '.'):
    enhancer = AdvancedDroneAudioEnhancer()
    success = enhancer.train_on_dregon_dataset(data_folder)

    if success:
        model_path = 'advanced_drone_enhancer.pkl'
        enhancer.save_model(model_path)
        print(f"Training completed successfully. Model saved as {model_path}")
        return enhancer
    else:
        print("Training failed")
        return None


def enhance_audio_file_advanced(input_file: str,
                                model_path: str = 'advanced_drone_enhancer.pkl',
                                output_file: Optional[str] = None) -> str:
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return None

    enhancer = AdvancedDroneAudioEnhancer()
    enhancer.load_model(model_path)

    audio_data, sample_rate = sf.read(input_file)

    if audio_data.ndim == 2:
        audio_data = np.mean(audio_data, axis=1)

    if sample_rate != enhancer.sr:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=enhancer.sr)

    chunk_duration = 30
    chunk_samples = int(chunk_duration * enhancer.sr)

    if len(audio_data) > chunk_samples:
        enhanced_chunks = []

        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i + chunk_samples]
            enhanced_chunk = enhancer.enhance_audio(chunk)
            enhanced_chunks.append(enhanced_chunk)

        enhanced = np.concatenate(enhanced_chunks)
    else:
        enhanced = enhancer.enhance_audio(audio_data)

    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_enhanced_advanced.wav"

    enhanced_normalized = enhanced / (np.max(np.abs(enhanced)) + 1e-10) * 0.95
    sf.write(output_file, enhanced_normalized, enhancer.sr)

    print(f"Audio enhancement completed. Output saved to {output_file}")
    return output_file


def analyze_enhancement_quality(original_file: str, enhanced_file: str, reference_file: str = None):
    original, sr = librosa.load(original_file, sr=16000)
    enhanced, _ = librosa.load(enhanced_file, sr=16000)

    min_len = min(len(original), len(enhanced))
    original = original[:min_len]
    enhanced = enhanced[:min_len]

    original_energy = np.mean(original ** 2)
    enhanced_energy = np.mean(enhanced ** 2)

    orig_stft = librosa.stft(original, n_fft=1024)
    enh_stft = librosa.stft(enhanced, n_fft=1024)

    orig_power = np.mean(np.abs(orig_stft) ** 2, axis=1)
    enh_power = np.mean(np.abs(enh_stft) ** 2, axis=1)

    freqs = librosa.fft_frequencies(sr=16000, n_fft=1024)

    suppression_ratio = enh_power / (orig_power + 1e-10)
    most_suppressed_freqs = freqs[suppression_ratio < 0.5]

    energy_change_db = 10 * np.log10(enhanced_energy / (original_energy + 1e-10))

    orig_centroid = np.mean(librosa.feature.spectral_centroid(S=np.abs(orig_stft), sr=16000))
    enh_centroid = np.mean(librosa.feature.spectral_centroid(S=np.abs(enh_stft), sr=16000))

    print(f"Enhancement analysis complete. Energy change: {energy_change_db:.1f} dB")
    print(f"Spectral centroid shift: {orig_centroid:.0f} Hz to {enh_centroid:.0f} Hz")

    if reference_file and os.path.exists(reference_file):
        reference, _ = librosa.load(reference_file, sr=16000)
        reference = reference[:min_len]

        correlation = np.corrcoef(enhanced, reference)[0, 1]
        print(f"Correlation with reference: {correlation:.3f}")


def batch_enhance_files(input_folder: str, output_folder: str = None,
                        model_path: str = 'advanced_drone_enhancer.pkl'):
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'enhanced')

    os.makedirs(output_folder, exist_ok=True)

    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = [f for f in os.listdir(input_folder)
                   if any(f.lower().endswith(ext) for ext in audio_extensions)]

    enhanced_files = []
    for i, filename in enumerate(audio_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_filename = f"enhanced_{os.path.splitext(filename)[0]}.wav"
        output_path = os.path.join(output_folder, output_filename)

        try:
            enhanced_file = enhance_audio_file_advanced(input_path, model_path, output_path)
            if enhanced_file:
                enhanced_files.append(enhanced_file)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Batch processing completed. Enhanced {len(enhanced_files)} of {len(audio_files)} files")
    return enhanced_files


def main():
    enhancer = train_advanced_enhancer()

    if enhancer:
        test_files = create_comprehensive_test()

        if test_files:
            for test_file in test_files:
                enhanced_file = enhance_audio_file_advanced(test_file)

                if enhanced_file:
                    analyze_enhancement_quality(test_file, enhanced_file, 'test_clean_reference.wav')

        print("System ready for audio enhancement operations")


if __name__ == "__main__":
    main()
    enhance_audio_file_advanced('noisy_audio 2.wav')