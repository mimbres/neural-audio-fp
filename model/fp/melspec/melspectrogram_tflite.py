# -*- coding: utf-8 -*-
"""melsprctrogram_tflite.py"""
import tensorflow as tf
from tensorflow.keras import Model
from kapre.time_frequency_tflite import STFTTflite, MagnitudeTflite
from kapre.time_frequency import ApplyFilterbank
import math


class Melspec_layer_lite(Model):
    """
    A wrapper class, based on TF-lite compatiable implementation:
        https://github.com/keunwoochoi/kapre
        
    Input:
        (B,1,T)
    Output:
        (B,C,T,1) with C=Number of mel-bins
    
    USAGE:
        
        See get_melspec_layer() in the below.

    """
    def __init__(
            self,
            #input_shape=(1, 8000),
            segment_norm=False,
            n_fft=1024,
            stft_hop=256,
            n_mels=256,
            fs=8000,
            dur=1.,
            f_min=300.,
            f_max=4000.,
            amin=1e-10, # minimum amp.
            dynamic_range=80.,
            name='Mel-spectrogram',
            trainable=False,
            **kwargs
            ):
        super(Melspec_layer_lite, self).__init__(name=name, trainable=False, **kwargs)
        
        self.mel_fb_kwargs = {
            'sample_rate': fs,
            'n_freq': n_fft // 2 + 1,
            'n_mels': n_mels,
            'f_min': f_min,
            'f_max': f_max,
            }
        self.stft_hop = stft_hop
        self.n_mels = n_mels
        self.amin = amin
        self.dynamic_range = dynamic_range
        self.segment_norm = segment_norm
        
        self.pad_l = n_fft // 2
        self.pad_r = n_fft // 2
        
        m_input_shape = (1, int(fs * dur) + self.pad_l + self.pad_r)
        self.m = tf.keras.Sequential(name=name)
        self.m.add(tf.keras.layers.InputLayer(input_shape=m_input_shape))
        self.m.add(
            STFTTflite(
                n_fft=n_fft,
                hop_length=stft_hop,
                pad_begin=False, # We do not use Kapre's padding, due to the @tf.function compatiability
                pad_end=False, # We do not use Kapre's padding, due to the @tf.function compatiability
                input_data_format='channels_first',
                output_data_format='channels_first'
                )
            )
        self.m.add(MagnitudeTflite())
        self.m.add(
            ApplyFilterbank(type='mel',
                            filterbank_kwargs=self.mel_fb_kwargs,
                            data_format='channels_first'
                            )
            )
        

    @tf.function
    def call(self, x):
        #input_shape = tf.shape(x)
        
        x = tf.pad(x, tf.constant([[0, 0], [0, 0], [self.pad_l, self.pad_r]])) # 'SAME' padding...
        
        x = self.m(x) + 0.1
        #x = tf.sqrt(x)
        
        x = tf.math.log(tf.maximum(x, self.amin)) / math.log(10)
        x = x - tf.reduce_max(x)
        x = tf.maximum(x, -1 * self.dynamic_range)
        if self.segment_norm:
            x = (x - tf.reduce_min(x) / 2) / tf.abs(tf.reduce_min(x) / 2 + 1e-10)
        x = tf.transpose(x, perm=(0,3,2,1))
        
        return x


    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[2] // self.stft_hop + 1 , self.n_mels)
    
    
def get_melspec_layer(cfg, trainable=False):
    fs = cfg['MODEL']['FS']
    dur = cfg['MODEL']['DUR']
    n_fft = cfg['MODEL']['STFT_WIN']
    stft_hop = cfg['MODEL']['STFT_HOP']
    n_mels = cfg['MODEL']['N_MELS']
    f_min = cfg['MODEL']['F_MIN']
    f_max = cfg['MODEL']['F_MAX']
    if cfg['MODEL']['FEAT'] == 'melspec':
        segment_norm = False
    elif cfg['MODEL']['FEAT'] == 'melspec_maxnorm':
        segment_norm = True
    else:
        raise NotImplementedError(cfg['MODEL']['FEAT'])
   
    l = Melspec_layer_lite(#input_shape=input_shape,
                           segment_norm=segment_norm,
                           n_fft=n_fft,
                           stft_hop=stft_hop,
                           n_mels=n_mels,
                           fs=fs,
                           dur=dur,
                           f_min=f_min,
                           f_max=f_max)
    l.trainable = trainable
    return l
                   
