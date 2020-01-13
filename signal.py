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