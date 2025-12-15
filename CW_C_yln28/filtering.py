import numpy as np
from scipy.signal import welch, spectrogram, filtfilt, butter, firwin, filtfilt, medfilt, savgol_filter
import matplotlib.pyplot as plt
import pywt


def estimate_band_from_psd(signals, fs, nperseg=1024, energy_frac=0.9, db_above_noise=6, plot=False):
    """
    signals: list or 2D array (n_signals, n_samples) or 1D array
    fs: sampling frequency
    energy_frac: use cumulative energy (e.g., 0.95 => band containing 95% energy)
    db_above_noise: alternative threshold method above noise floor in dB
    Returns: (f, Pxx_mean), band_energy_cutoffs (f_low, f_high), band_noise_cutoffs (f_low2,f_high2)
    """
    arr = np.atleast_2d(signals)
    if arr.shape[0] > arr.shape[1] and arr.shape[1] == fs:  # crude guess - ensure shape (n_signals, n_samples)
        pass
    # compute PSD for each and average
    psds = []
    freqs = None
    for x in arr:
        f, Pxx = welch(x, fs=fs, nperseg=nperseg)
        psds.append(Pxx)
        freqs = f
    Pxx_mean = np.mean(psds, axis=0)

    # cumulative energy method
    cum = np.cumsum(Pxx_mean)
    cum = cum / cum[-1]
    # find band that covers [ (1-energy_frac)/2 , (1+energy_frac)/2 ] of energy centered
    # simpler: find lowest freq where cum> (1-c.energy_frac) and highest where cum>energy_frac
    low_idx = np.searchsorted(cum, (1-energy_frac)/2) if (1-energy_frac)/2>0 else 0
    high_idx = np.searchsorted(cum, 1-(1-energy_frac)/2) if (1-energy_frac)/2>0 else np.searchsorted(cum, energy_frac)
    f_low_energy, f_high_energy = freqs[low_idx], freqs[min(high_idx, len(freqs)-1)]

    # noise-floor threshold method (dB)
    # estimate noise floor from top 10% of freqs (near Nyquist)
    noise_region = freqs > (0.8 * (fs/2))
    noise_floor = np.median(Pxx_mean[noise_region])
    Pxx_db = 10*np.log10(Pxx_mean + 1e-20)
    noise_db = 10*np.log10(noise_floor + 1e-20)
    # find contiguous region where Pxx_db > noise_db + db_above_noise
    mask = Pxx_db > (noise_db + db_above_noise)
    # get first and last True indices
    if mask.any():
        idxs = np.where(mask)[0]
        f_low_db, f_high_db = freqs[idxs[0]], freqs[idxs[-1]]
    else:
        f_low_db, f_high_db = freqs[0], freqs[-1]

    # Plot for inspection
    if plot:
        plt.figure(figsize=(8,4))
        plt.semilogy(freqs, Pxx_mean, label='Mean PSD')
        plt.axvline(f_low_energy, color='C1', linestyle='--', label=f'energy low {f_low_energy:.1f} Hz')
        plt.axvline(f_high_energy, color='C1', linestyle='--', label=f'energy high {f_high_energy:.1f} Hz')
        plt.axvline(f_low_db, color='C2', linestyle=':', label=f'db low {f_low_db:.1f} Hz')
        plt.axvline(f_high_db, color='C2', linestyle=':', label=f'db high {f_high_db:.1f} Hz')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.legend()
        plt.grid(True)
        plt.title('Average PSD with estimated useful bands')
        plt.show()

    return freqs, Pxx_mean, (f_low_energy, f_high_energy), (f_low_db, f_high_db)


def plot_spectrogram(x, fs, nperseg=256, noverlap=200):
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    plt.figure(figsize=(8,4))
    plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-20), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='PSD (dB)')
    plt.ylim(0, fs/2)
    plt.title('Spectrogram (dB)')
    plt.show()


def bandpass_300_3000(x, fs=25000, order=4):
    """
    300–3000 Hz Butterworth bandpass filter.
    Zero-phase using filtfilt.
    """
    nyq = fs * 0.5
    low = 300 / nyq
    high = 3000 / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)


def lowpass_butter(x, cutoff, fs=25000, order=4):
    b, a = butter(order, cutoff, btype='low', fs=fs)
    y = filtfilt(b, a, x)
    return y


def detrend(x):
    # simple linear detrend; replace with your preferred method
    t = np.arange(len(x))
    p = np.polyfit(t, x, 1)
    return x - (p[0]*t + p[1])


def swt_denoise(x, wavelet='sym8', level=None, method='soft'):
    """
    Stationary Wavelet Transform denoising.
    - wavelet: 'sym8','db8','coif4' are good starts
    - level: number of levels; if None choose floor(log2(len(x)))-1 but clip to 1..6
    - method: 'soft' or 'hard'. soft usually better.
    """
    n = len(x)
    if level is None:
        level = int(np.floor(np.log2(n))) - 1
    level = max(1, min(level, 6))  # sensible bounds

    # SWT decomposition (coeffs: [(cA_n, cD_n), ..., (cA1, cD1)])
    coeffs = pywt.swt(x, wavelet, level=level, start_level=0, axis=-1)

    # estimate noise sigma from finest-scale detail coefficients (last tuple's cD)
    # Use median absolute deviation (MAD)
    cA_n, cD_n = coeffs[-1]
    sigma = np.median(np.abs(cD_n)) / 0.6745 + 1e-16

    # universal threshold (Donoho)
    uthresh = sigma * np.sqrt(2 * np.log(n))

    # Threshold detail coefficients at each level
    new_coeffs = []
    for cA, cD in coeffs:
        # soft thresholding
        cD_thresh = pywt.threshold(cD, uthresh, mode=method)
        new_coeffs.append((cA, cD_thresh))

    # Reconstruct
    x_rec = pywt.iswt(new_coeffs, wavelet)
    return x_rec


def fir_bandpass_filter(x, fs, f_low, f_high, numtaps=401, window='hamming'):
    nyq = fs / 2.0
    if f_low <= 0 and f_high >= nyq:
        return x  # nothing to do
    if f_low <= 0:
        taps = firwin(numtaps, cutoff=f_high/nyq, window=window)
    elif f_high >= nyq:
        taps = firwin(numtaps, cutoff=f_low/nyq, pass_zero=False, window=window)
    else:
        taps = firwin(numtaps, [f_low/nyq, f_high/nyq], pass_zero=False, window=window)
    # padlen must be > 3*(len(taps)-1), filtfilt handles internally but include padlen param for safety
    y = filtfilt(taps, [1.0], x, padlen=min(len(x)-1, 3*(len(taps)-1)))
    return y


def median_clean(x, kernel_size=31):
    # kernel_size must be odd
    return x - medfilt(x, kernel_size=kernel_size)


def savgol_smooth(x, window_length=31, polyorder=3):
    if window_length >= len(x):
        window_length = len(x)-1 if (len(x)-1)%2==1 else len(x)-2
    if window_length < 5:
        window_length = 5
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(x, window_length=window_length, polyorder=polyorder)


def denoise_pipeline(x, fs=25000,
                     do_detrend=True,
                     do_wavelet=False,
                     swt_wavelet='sym8', swt_level=4,
                     numtaps=401, 
                     do_bandpass=False, bandpass=None, # (f_low,f_high) in Hz or None
                     do_median=True, median_k=201,
                     do_savgol=True, sg_window=31, sg_poly=4):
    
    x_filt = x.copy()
    if do_detrend:
        x_filt = detrend(x_filt)
    
    # Gentle bandpass
    if do_bandpass:
        f_low, f_high = bandpass
        x_filt = fir_bandpass_filter(x_filt, fs, f_low, f_high, numtaps=numtaps)

        # x_filt = bandpass_300_3000(x_filt, order=4)

    # Wavelet denoise (SWT)
    if do_wavelet:
        x_filt = swt_denoise(x_filt, wavelet=swt_wavelet, level=swt_level)
    
    # median
    if do_median:
        x_filt = median_clean(x_filt, kernel_size=median_k)
   
    # savgol
    if do_savgol:
        x_filt = savgol_smooth(x_filt, window_length=sg_window, polyorder=sg_poly)

    return x_filt


def test():
    from main import import_matdata
    matdata = import_matdata()
    freqs, Pxx_mean, band_energy, band_db = estimate_band_from_psd(matdata["D5"]["d"], fs=25e3)
    print('Energy band:', band_energy)
    print('DB threshold band:', band_db)

    # plot_spectrogram(matdata["D1"]["d"], fs=25e3)
    
    # savgol for D1-3

    # usage
    # y = lowpass_butter(matdata["D4"]["d"], cutoff=2000, fs=25000)   # lowpass butter 2000 for D4
    
    data = matdata["D5"]["d"]
    # data = data / np.max(data)  # filter peaks by prominence, 0.08 should do. min height, 0.05. 
    y = denoise_pipeline(data, bandpass=band_db, do_savgol=True)
    
    plt.plot(y)
    plt.show()


if __name__ == "__main__":
    test()