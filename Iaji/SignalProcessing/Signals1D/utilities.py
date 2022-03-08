"""
@author : Iyad Suleiman
"""
#%%
#imports
import numpy
from scipy import signal
#%%
def NyquistReconstruct(x, fs, upsampling_factor, n_low_pass=501):
    """
    This function reconstruct the original signal, given a sampled version of it, 
    namely 'x', sampled at sampling rate 'fs', assuming the underlying signal
    was sampling respecting the Nyquist criterion, i.e., its bandwidth is lower
    than the Nyquist frequency fs/2. The reconstruction system consists in two transformations:
        
        - Interpolation, with upsampling factor equal to 'upsampling_factor'
        - Low-pass (FIR) filtering with cutoff frequency fs/2
    
    According to Nyquist's theorem, the result is close to the underlying signal, before sampling.
    The quality of the reconstruction depends on the upsampling factor and on the 
    number of low-pass filter coefficients.
    
    INPUTS
    ---------------
        x : 1D array-like of float
            Sampled signal
        fs : float (>0)
            Sampling rate
        upampling_factor : int (>0)
            Ratio between the sampling rates of the reconstructed and sampled signals
        n_low_pass : int (>0)
            Number of low-pass filter coefficients
    """
    n_samples = len(x)
    b_low_pass = signal.firwin(numtaps=n_low_pass, cutoff=fs/2, nyq=fs*upsampling_factor/2)
    x_reconstructed = signal.filtfilt(b_low_pass, 1, signal.resample(x, num=n_samples*upsampling_factor))
    return x_reconstructed
#%%
def detectPhaseIQ(x, fs, f_low_pass, n_low_pass=501, n_hilbert=None):
    """
    This function estimates the instantaneous phase of an input signal, supposedly centered
    at a well-defined Fourier frequency, by demodulating it with two orthogonal sinusoidal functions
    at that frequency, yielding the signals 'i' and 'q'. The phase is then computed as
    
        phase = numpy.unwrap(numpy.angle(i + 1j*q))
        
    where 1j is the imaginary unit in Python language. The user can input the reference signal,
    from which the reference sinusoidal functions are computed, as a second column to the array x; 
    else, the reference frequency is extracted from x 
    and the reference sinusoidal functions are constructed digitally. 
    
    INPUTS
    ------------
        x : array-like of floats
            The input signal. If x is a 2-D array, then the first column
            is interpreted as the input signal samples, and the second column
            is interpreted as the reference signal samples.
        fs : float (>0)
            Sampling rate
        n_low_pass : int (>0)
            Number of coefficients of the demodulation low-pass filter
        n_hilbert : int (>0)
            Number of coefficients of the Hilbert transformer that yields the orthogonal
            reference signal
    
    OUTPUTS
    ------------
        The instantaneous phase of the input signal x[:, 0]
    """
    #Extract the input signal
    input_signal = []
    reference_signal = []
    #See if the reference signal has been provided by the user
    if (len(numpy.shape(x))==2):
        input_signal = x[:, 0]
        reference_signal = x[:, 1]
        reference_signal /= numpy.std(reference_signal)*numpy.sqrt(2)
    #Else, construct it by estimating the central frequency of the reference signal from the input signal
    else:
        input_signal = x
        t = numpy.array(range(len(input_signal)))*1/fs#time [same units as 1/fs]
        #Compute the central frequency from the periodogram of the input signal
        frequency, PSD = signal.periodogram(x=input_signal, fs=fs)
        #Estimate the central frequency as the maximum of the periodogram
        f_r = numpy.atleast_1d(frequency[numpy.where(abs(PSD)==max(abs(PSD)))])
        f_r = f_r[int(len(f_r)/2)]        
        #Construct the reference signal
        reference_signal = numpy.cos(2*numpy.pi*f_r*t)
    #Construct the orthogonal reference signal
    reference_signal_orthogonal = []
    if n_hilbert: #if the number of coefficients of the Hilbert transformer has been specified
        reference_signal_orthogonal = numpy.imag(signal.hilbert(x=reference_signal, N=n_hilbert))
        #Discard the first samples due to the filter's transient response
        n_samples_excluded = int(numpy.ceil(n_hilbert/2))
        input_signal = input_signal[n_samples_excluded-1:]
        reference_signal = reference_signal[n_samples_excluded-1:]
        reference_signal_orthogonal = reference_signal_orthogonal[n_samples_excluded-1:]
    else:
        reference_signal_orthogonal = -numpy.gradient(reference_signal)
    reference_signal_orthogonal /= numpy.std(reference_signal_orthogonal)*numpy.sqrt(2)
    #Construct the low-pass filter
    b_low_pass = signal.firwin(numtaps=n_low_pass, cutoff=f_low_pass, nyq=fs/2)
    #Demodulate the input signal
    i = signal.filtfilt(b_low_pass, 1, input_signal*reference_signal)
    q = signal.filtfilt(b_low_pass, 1, input_signal*reference_signal_orthogonal)
    n_samples_excluded = int(numpy.ceil(n_low_pass/2))
    i = i[n_samples_excluded-1:]
    i = i[::int(fs/(2*f_low_pass))]
    q = q[n_samples_excluded-1:]
    q = q[::int(fs/(2*f_low_pass))]
    instantaneous_phase = numpy.unwrap(numpy.angle(i+1j*q))
    return instantaneous_phase
#%%
#Define a function that computes the "moving" root mean squared of an input signal
#with windows of chosen length (in number of samples)
def movingAverage(x, Fs, sampling_times):
    """
    This function computes the moving average of the input signal 'x', sampled
    with sampling frequency 'Fs', by averaging within the time intervals specified
    by the sequence 'sampling_times'.
    
    INPUTS
    -------------
    x : array-like of float
        Input signal [a.u.]
    Fs : float (>0)
        Sampling frequency of the input signal  [Hz]
    sampling_times : array-like of float
        Sampling instants for averaging
        
    OUTPUTS
    --------------
    x_mean : array-like of float
        Moving mean of the input signal 
    """
    n_samples = len(x)
    n_points = len(sampling_times)
    x_mean = numpy.zeros(n_points-1)
    time = numpy.array(range(n_samples))/Fs
    for j in range(n_points-1):
        if sampling_times[j+1] > time[-1]:
            break
        interval = numpy.where(numpy.logical_and(time>=sampling_times[j], time<sampling_times[j+1]))
        x_mean[j] = numpy.mean(x[interval])
    return x_mean
       
        
    
    
    