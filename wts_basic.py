import numpy as np
from pathlib import Path
from scipy.io import wavfile


class LinearInterpolator():
    def __call__(self, values, index):
        """Interpolate linearly between values closest to index."""
        # Closest index value smaller or equal to index
        low = int(index)

        # Closest index value larger than index
        high = int(np.ceil(index))

        # Handle integer index
        if low == high:
            return values[low]

        # Return the weighted sum
        return (index - low) * values[high %
                                      values.shape[0]] + (high - index) * values[low]


class ZeroOrderInterpolator():
    def __call__(self, values, index):
        """
        Return the value in values at the largest integer index
        smaller or equal to index.
        """
        return values[int(index)]


class WavetableOscillator:
    """
    Oscillator generates samples based on current frequency.
    If frequency is 0, no samples are produced.

    WavetableOscillator uses wavetable synthesis to generate each sample.
    To this end, it needs a wavetable, the sampling rate 
    and the interpolation method used.
    """

    def __init__(self, wavetable, sampling_rate, interpolator):
        self.wavetable = wavetable
        self.sampling_rate = sampling_rate
        self.interpolator = interpolator
        self.wavetable_index = 0.0
        self.__frequency = 0

    def fill(self, audio_block, from_index=0, to_index=-1):
        """Fill audio_block in [from_index, to_index) range."""
        for i in range(from_index, to_index % audio_block.shape[0]):
            audio_block[i] = self.get_sample()
        return audio_block

    def get_sample(self):
        """Return 1 oscillator sample."""
        sample = self.interpolator(self.wavetable, self.wavetable_index)
        self.wavetable_index = (
            self.wavetable_index + self.wavetable_increment) % self.wavetable.shape[0]
        return sample

    @property
    def frequency(self):
        return self.__frequency

    @frequency.setter
    def frequency(self, value):
        self.__frequency = value
        self.wavetable_increment = self.wavetable.shape[0] * \
            self.frequency / self.sampling_rate
        if self.frequency <= 0:
            self.wavetable_index = 0.0


class Voice:
    """
    Main class to generate sound.

    1 Voice object corresponds to 1 pitch played.

    Properties
    ----------
    oscillators: list
        list containing the oscillators that generate sound
    gain: float
        amplitude in dBFS of the generated signal
    """

    def __init__(self, sampling_rate, gain=-20):
        self.sampling_rate = sampling_rate
        self.gain = gain
        self.oscillators = []

    def synthesize(self, frequency, duration_seconds):
        """
        Generate duration_seconds of samples at the 
        specified frequency.
        """
        buffer = np.zeros((duration_seconds * self.sampling_rate,))
        if np.isscalar(frequency):
            frequency = np.ones_like(buffer) * frequency

        for i in range(len(buffer)):
            for oscillator in self.oscillators:
                oscillator.frequency = frequency[i]
                buffer[i] += oscillator.get_sample()
        amplitude = 10 ** (self.gain / 20)
        buffer *= amplitude
        buffer = fade_in_out(buffer)
        return buffer


def fade_in_out(signal, fade_length=1000):
    """
    Apply a half-cosine window to first and 
    last fade_length samples of signal.
    """
    fade_in_envelope = (1 - np.cos(np.linspace(0, np.pi, fade_length))) * 0.5
    fade_out_envelope = np.flip(fade_in_envelope)

    # Handle 2-channel audio
    if signal.ndim == 2:
        fade_in_envelope = fade_in_envelope[:, np.newaxis]
        fade_out_envelope = fade_out_envelope[:, np.newaxis]

    # Apply fade-in
    signal[:fade_length, ...] = np.multiply(
        signal[:fade_length, ...], fade_in_envelope)

    # Apply fade-out
    signal[-fade_length:, ...] = np.multiply(
        signal[-fade_length:, ...], fade_out_envelope)

    return signal


def generate_wavetable(length, f):
    """
    Generate a wavetable of specified length using 
    function f(x) where x is phase.
    Period of f is assumed to be 2 pi.
    """
    wavetable = np.zeros((length,), dtype=np.float32)
    for i in range(length):
        wavetable[i] = f(2 * np.pi * i / length)
    return wavetable


def output_wavs(signal, name, sampling_rate, table):
    """Save the signal and wave table under specified names."""
    output_dir = Path('wavetable-synthesis-python')
    output_dir.mkdir(parents=True, exist_ok=True)

    wavfile.write(
        output_dir /
        f'{name}_table.wav',
        sampling_rate,
        table.astype(
            np.float32))
    wavfile.write(
        output_dir /
        f'{name}.wav',
        sampling_rate,
        signal.astype(
            np.float32))


def gaussian_mixture(x):
    """Sum of 5 Gaussians roughly in [0,2pi) range."""
    return np.exp(-3 * (x - 1)**2) \
        - 0.4 * np.exp(-3 * (x - 2.3)**2) \
        + 0.8 * np.exp(-10 * (x - 3.3)**2) \
        - np.exp(-7 * (x - 4.5)**2) \
        + 0.3 * np.exp(-2 * (x - 5)**2)


def generate_gaussians_table(wavetable_size):
    """Generate the wave table with 5 Gaussians."""
    gaussians_table = generate_wavetable(wavetable_size, gaussian_mixture)

    # Subtract the DC component
    gaussians_table -= np.mean(gaussians_table)

    # Smooth out at the edges so that 
    # the wave table starts and ends around 0.0
    gaussians_table = fade_in_out(gaussians_table, 5)
    return gaussians_table


def sawtooth_waveform(x):
    """Sawtooth with period 2 pi."""
    return (x + np.pi) / np.pi % 2 - 1


def square_waveform(x):
    """Square waveform with period 2 pi."""
    return np.sign(np.sin(x))


def main():
    # Global parameters
    sampling_rate = 44100
    wavetable_size = 64

    # Create a mono synth
    synth = Voice(sampling_rate, gain=-20)

    ### Sine generation ###
    sine_table = generate_wavetable(wavetable_size, np.sin)
    # Add an oscillator
    synth.oscillators += [
        WavetableOscillator(
            sine_table,
            sampling_rate,
            LinearInterpolator())]
    # Synthesize sound
    sine = synth.synthesize(frequency=440, duration_seconds=5)
    # Save the output
    output_wavs(sine, 'sine', sampling_rate, sine_table)

    ### Sawtooth generation ###
    sawtooth_table = generate_wavetable(wavetable_size, sawtooth_waveform)
    synth.oscillators[0] = WavetableOscillator(
        sawtooth_table, sampling_rate, LinearInterpolator())
    sawtooth_signal = synth.synthesize(frequency=440, duration_seconds=5)
    output_wavs(sawtooth_signal, 'sawtooth', sampling_rate, sawtooth_table)

    sawtooth880 = synth.synthesize(frequency=880, duration_seconds=5)
    output_wavs(sawtooth880, 'sawtooth880', sampling_rate, sawtooth_table)

    ### Multi-cycle waveform generation ###
    square_table = generate_wavetable(wavetable_size, square_waveform)
    multi_cycle_table = np.concatenate(
        (sine_table, square_table, sawtooth_table))
    synth.oscillators[0] = WavetableOscillator(
        multi_cycle_table, sampling_rate, LinearInterpolator())
    # Frequency is divided by 3 because we concatenated 3 tables
    multi_cycle = synth.synthesize(frequency=330 / 3, duration_seconds=5)
    output_wavs(multi_cycle, 'multi_cycle', sampling_rate, multi_cycle_table)

    ### Gaussian mixture generation ###
    gaussians_table = generate_gaussians_table(wavetable_size)
    synth.oscillators[0] = WavetableOscillator(
        gaussians_table, sampling_rate, LinearInterpolator())
    gaussians_waveform = synth.synthesize(frequency=110, duration_seconds=5)
    output_wavs(
        gaussians_waveform,
        'gaussians',
        sampling_rate,
        gaussians_table)

    ### Continuous frequency control ###
    # We want to generate and exponentially swept frequency vector.
    # Frequency will start at min_frequency, reach max_frequency after
    # half of the duration and then fall back to min_frequency.
    duration = 20
    min_frequency = 20
    max_frequency = 3000

    # Calculate the base of the exponent
    base = (max_frequency / min_frequency) ** (1 /
                                               (duration // 2 * sampling_rate))

    # Calculate the exponential frequency sweep on the rising slope
    instantaneous_frequency_half = min_frequency * \
        base ** np.arange(0, duration // 2 * sampling_rate, 1)

    # Make the falling slope the reverse of the first slope
    instantaneous_frequency = np.concatenate(
        (instantaneous_frequency_half, np.flip(instantaneous_frequency_half)))

    # Add tiny oscillations around the intantaneous frequency
    instantaneous_frequency += np.multiply(instantaneous_frequency,
                                           np.random.default_rng().uniform(-0.1, 0.1, size=instantaneous_frequency.shape))

    # Synthesize on a sample-by-sample basis and output
    signal_with_varying_frequency = synth.synthesize(
        frequency=instantaneous_frequency, duration_seconds=duration)
    output_wavs(
        signal_with_varying_frequency,
        'instantaneous_frequency',
        sampling_rate,
        gaussians_table)


if __name__ == '__main__':
    main()