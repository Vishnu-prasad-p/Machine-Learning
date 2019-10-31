#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:59:37 2018

@author: apple
"""
#%%
import numpy as np

import matplotlib.pyplot as plot

 

# Get x values of the sine wave

time        = np.arange(0, 10, 0.1);

 
f=40
# Amplitude of the sine wave is sine of a variable like time

amplitude   = np.sin(2*np.pi*f*time)

 

# Plot a sine wave using time and amplitude obtained for the sine wave

plot.plot(time, amplitude)

 

# Give a title for the sine wave plot

plot.title('Sine wave')

 

# Give x axis label for the sine wave plot

plot.xlabel('Time')

 

# Give y axis label for the sine wave plot

plot.ylabel('Amplitude = sin(time)')

 

plot.grid(True, which='both')

 

plot.axhline(y=0, color='k')

 

plot.show()

 

# Display the sine wave

plot.show()

#%%
import matplotlib.pyplot as plt
import numpy as np


Fs = 8000
f = 200
sample = 8000
x = np.arange(0, 10, 0.1);
y = np.sin(2 * np.pi * f * x )
plt.plot(x, y)
plt.xlabel('sample(n)')
plt.ylabel('voltage(V)')
plt.show()

#%%
import matplotlib.pyplot as plt # For ploting
import numpy as np # to work with numerical data efficiently

fs = 10# sample rate 
f = 40 # the frequency of the signal

x = np.arange(fs) # the points on the x axis for plotting
# compute the value (amplitude) of the sin wave at the for each sample
y = [ np.sin(2*np.pi*f * (i/fs)) for i in x]

# showing the exact location of the smaples
#plt.stem(x,y, 'r', )
plt.plot(x,y)
#%%
import matplotlib.pyplot as plt # For ploting
import numpy as np # to work with numerical data efficiently
from scipy import signal

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd,
                               X_even + factor[N / 2:] * X_odd])

f = [200,401,800] # sample rate 
f1 = 20 # the frequency of the signal
f2 = 40
f3 = 100
f4 = 200
win = signal.hann(50)
for fs in f:
    x = np.arange(fs) # the points on the x axis for plotting
    # compute the value (amplitude) of the sin wave at the for each sample
    y1 = [ np.sin(2*np.pi*f1 * (i/fs)) for i in x]
    y2 = [ np.sin(2*np.pi*f2 * (i/fs)) for i in x]
    y3 = [ np.sin(2*np.pi*f3 * (i/fs)) for i in x]
    y4 = [ np.sin(2*np.pi*f4 * (i/fs)) for i in x]
    
    y=np.array(y1)+np.array(y2)+np.array(y3)+np.array(y4)
    
    #this instruction can only be used with IPython Notbook. 
    #% matplotlib inline
    # showing the exact location of the smaples
    #plt.stem(x,y, 'r', )
    
    plt.subplot(511)
    plt.plot(x,y1)
    plt.title('Sine wave')
    
    plt.subplot(512) 
    plt.plot(x,y2)
    
    plt.subplot(513) 
    plt.plot(x,y3)
    plt.ylabel('Amplitude = sin(time)')
    
    plt.subplot(514) 
    plt.plot(x,y4)
    
    plt.subplot(515) 
    plt.plot(x,y)
    plt.xlabel('Time')
    
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.figure()
    
filtered = np.convolve(y, win, mode='full')
plt.plot(filtered)
plt.figure()

ff=np.fft.fft(y)
plt.plot(ff)
plt.figure()

f1t=FFT(y)
plt.plot(f1t)
plt.figure()

f2t=DFT_slow(y)
plt.plot(f2t)

plt.figure()



