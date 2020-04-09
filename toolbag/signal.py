import numpy as np

def trim_signal(signal, times, window):
    '''Trim signal to time window.
    
    Parameters
    ----------
    signal : ndarray (..., n_times)
    times : 1d array (n_times,)
    window : tuple (ini_t, end_t)

    Returns
    -------
    ndarray with trimmed signal.
    '''

    ini_ix, end_ix = time2ix(times, window)
    return signal[..., ini_ix:end_ix]

def time2ix(times, t):
    ''' Returns index of the timepoint in `times` closest to `t`.

    Parameters
    ----------
    times : 1d array
        Refere

    t : float or 1d array

    Returns
    -------
    Index(es) of closest timepoint(s).
    '''

    def aux(times, t):
        return np.abs(np.array(times) - t).argmin()

    if hasattr(t, '__iter__'):
        return np.array([aux(times, tt) for tt in t])
    return aux(times, t)

def zscore_signal(signal, times, window):
    baseline = toolbag.signal.trim_signal(signal, times, window)
    mean = np.mean(baseline, axis=1)
    std = np.std(baseline, axis=1)

    z_signal = (signal - mean[:, np.newaxis]) / std[:, np.newaxis]
    return z_signal

def tf_morlet(signal, freqs, n_cycles, srate):
    '''Computes time-frequency decomposition using Morlet waves.
    
    Parameters
    ----------
    signal : array (n_trials, n_times)
        Epochs of one channels.
    
    freqs : array (n_freqs,)
        The frequencies.
    
    n_cycles : int or array (n_freqs,)
        Number of cycles in the morlet wave
        
    srate : float
        Sampling frequency.
    
    Returns
    -------
    array (n_trials, n_freqs, n_times)
        Power of signal.


    Example
    -------
        ch = 'e30'
        epochs_ch = np.squeeze(np.array(epochs_mne.copy().pick(ch)))
        times = epochs.times * 1000

        fs = 1000
        min_n_cycle = 1
        max_n_cycle = 12
        min_freq = 7
        max_freq = 200
        n_freqs = 100
        freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)
        n_cycles = np.linspace(min_n_cycle, max_n_cycle, len(freqs))

        power = tf_morlet(epochs_ch, freqs, n_cycles, fs)
        power = mne.baseline.rescale(power, times, (-250, -50), mode='logratio', copy=True);
        power_avg = np.mean(power, axis=0)
    '''
    
    n_trials, n_data = signal.shape
    power = np.zeros((len(freqs), n_trials, n_data))
    
    wave_times = np.r_[np.arange(-1, 1, 1/srate), [1]]
    half_wave = int((len(wave_times)-1)/2)
    n_kernel = wave_times.shape[0]
    n_conv = n_kernel + n_data - 1
    print(n_conv)
    
    for i, f in enumerate(freqs):
        if hasattr(n_cycles, '__len__'):
            assert (len(n_cycles) == len(freqs))
            std = n_cycles[i] / (2*np.pi * f)
        else:
            std = n_cycles / (2*np.pi * f)
            
        gauss_window = np.exp((-(wave_times ** 2)) / (2 * std**2))
        sine_wave = np.exp(1j * 2*np.pi * f * wave_times)
        wavelet = sine_wave * gauss_window
        wavelet_F = np.fft.fft(wavelet, n_conv)
        wavelet_F = wavelet_F / np.max(wavelet_F)
        
        signal_F = np.fft.fft(signal, n=n_conv, axis=1)
        timef = np.fft.ifft(wavelet_F * signal_F)
        timef = timef[:, half_wave + 1 : -half_wave + 1]
        power[i, ...] = np.abs(timef) ** 2
        
    power = np.swapaxes(power, 0, 1)
    return power