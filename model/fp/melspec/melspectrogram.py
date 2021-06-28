# -*- coding: utf-8 -*-
"""melsprctrogram.py"""    
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda, Permute
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank
import math


class Melspec_layer(Model):
    """
    A wrapper class, based on the implementation:
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
            input_shape=(1, 8000),
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
        super(Melspec_layer, self).__init__(name=name, trainable=False, **kwargs)
        
        self.mel_fb_kwargs = {
            'sample_rate': fs,
            'n_freq': n_fft // 2 + 1,
            'n_mels': n_mels,
            'f_min': f_min,
            'f_max': f_max,
            }
        self.n_fft = n_fft
        self.stft_hop = stft_hop
        self.n_mels = n_mels
        self.amin = amin
        self.dynamic_range = dynamic_range
        self.segment_norm = segment_norm
        
        # 'SAME' Padding layer
        self.pad_l = n_fft // 2
        self.pad_r = n_fft // 2
        self.padded_input_shape = (1, int(fs * dur) + self.pad_l + self.pad_r)
        self.pad_layer = Lambda(
            lambda z: tf.pad(z, tf.constant([[0, 0], [0, 0],
                                             [self.pad_l, self.pad_r]]))
            )
        
        # Construct log-power Mel-spec layer
        self.m = self.construct_melspec_layer(input_shape, name)

        # Permute layer
        self.p = tf.keras.Sequential(name='Permute')
        self.p.add(Permute((3, 2, 1), input_shape=self.m.output_shape[1:]))
        
        super(Melspec_layer, self).build((None, input_shape[0], input_shape[1]))
        
        
    def construct_melspec_layer(self, input_shape, name):
        m = tf.keras.Sequential(name=name)
        m.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        m.add(self.pad_layer)
        m.add(
            STFT(
                n_fft=self.n_fft,
                hop_length=self.stft_hop,
                pad_begin=False, # We do not use Kapre's padding, due to the @tf.function compatiability
                pad_end=False, # We do not use Kapre's padding, due to the @tf.function compatiability
                input_data_format='channels_first',
                output_data_format='channels_first')
            )
        m.add(
            Magnitude()
            )
        m.add(
            ApplyFilterbank(type='mel',
                            filterbank_kwargs=self.mel_fb_kwargs,
                            data_format='channels_first'
                            )
            )
        return m
        

    @tf.function
    def call(self, x):        
        x = self.m(x) + 0.06
        #x = tf.sqrt(x)
        
        x = tf.math.log(tf.maximum(x, self.amin)) / math.log(10)
        x = x - tf.reduce_max(x)
        x = tf.maximum(x, -1 * self.dynamic_range)
        if self.segment_norm:
            x = (x - tf.reduce_min(x) / 2) / tf.abs(tf.reduce_min(x) / 2 + 1e-10)
        return self.p(x) # Permute((3,2,1))

    
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
    
    input_shape = (1, int(fs * dur))
    l = Melspec_layer(input_shape=input_shape,
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
                        