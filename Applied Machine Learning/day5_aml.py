#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 14:22:14 2018

@author: apple
"""
import wave
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

(fs,x)=read('Yamaha-TG100-Kalimba-C5.wav')
plt.plot(x[:,0])
plt.figure()
plt.plot(x[:,1])



