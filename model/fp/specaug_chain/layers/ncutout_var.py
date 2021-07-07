# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" ncutout_var.py 

SpecNCutout: Batch-wise SpecAugment + N-CutOut layers for augmentation in 
spectral domain for GPUs.

• variable number of holes for cutout
• hole filler types ['zeros', 'random']
• compatiable with TF2.x and @tf.function decorator
• implementation based on tf.Variable.assign


USAGE:

    spec_ncutout_layer = SpecNCutout(prob=0.5,
                                     n_holes=3,
                                     hole_fill='random')
    m = (your method to get spectrogram here...)
    m_aug = spec_ncutout_layer(m) 
    
For more details, see test() in the below

References:
    • original NumPy implementation of https://arxiv.org/abs/1708.04552       
        https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        
"""
import tensorflow as tf
import numpy as np              # for test()
import matplotlib.pyplot as plt # for test()
from io import BytesIO          # for test()


class SpecNCutout(tf.keras.layers.Layer):
    """ 
    SpecNCutout based on SpecAugment and CutOut: implementation based on
    tf.Variable.assign.
    
    Arguments
    ---------        
    • prob: (tf.Float)
        probability (0-1) of cutout activity. If prob=1.0 (default), always
        output cut-out spectrogram.
    • n_holes: (int)
        Number of random hole masks to create.
    • uniform_mask: (bool)
        If True (default), apply a uniform mask to the batch. 
    • hole_fill: (str) or [(float), (float)]
        Values to fill the holes with. Default is 'min'.
        - 'min' fills with minimum magnitude of input spectrogram.
        - 'zeros' fills with zeros.
        - 'random' fills with random values within range of min and max of input.
        - [min_mag, max_mag] fills with random values within the range.
    • hole_config: [(int),(int),(int),(int)]
        Configures the range of hole mask size by [min_width, max_width,
        min_height, max_height]. If [None,None,None,None] (default), 1/10 of
        the input length will be set as minimum, and 1/2 of input length will
        be set as maximum.
    • internal_variable: (bool) 
        Set False if using as a component of specaug_chainer class that creates
        tf.Variable. Default is True.

    Input
    -----
    (tf.Float) 4D tensor variable with shape (B,H,W,C), equivalent with
        (Batch,Freq,Time,1).
                  
    """
    def __init__(self,
                  prob=1.0,
                  n_holes=1,
                  uniform_mask=True,
                  hole_fill='min',
                  hole_config=[None,None,None,None],
                  **kwargs):
        super(SpecNCutout, self).__init__()
        self.prob = prob
        self.n_holes = n_holes
        self.uniform_mask = uniform_mask
        self.hole_fill, self.filler_min, self.filler_max = None, None, None
        if hole_fill in ['min', 'zeros', 'random']:
            self.hole_fill = hole_fill
            self.filler_min = None
            self.filler_max = None
        elif type(hole_fill)==list and len(hole_fill)==2:
            self.hole_fill = 'random_with_range'
            self.filler_min = hole_fill[0]
            self.filler_max = hole_fill[1]
        else:
            raise NotImplementedError(hole_fill)
            
        self.hole_minw, self.hole_maxw, self.hole_minh, self.hole_maxh = hole_config


    def build(self, input_shape):
        if tf.__version__ >= "2.0":
            self.varx = tf.Variable(
                initial_value=tf.zeros(input_shape), validate_shape=True,
                #shape=input_shape, # <-- TF bug here! (None shape is not translated from None input_shape)
                #shape=(None, input_shape[1], input_shape[2], input_shape[3]),
                shape=input_shape,
                trainable=False,  dtype=tf.float32)
        else:
            self.varx = tf.Variable(
                initial_value=tf.zeros(input_shape),  validate_shape=True,
                #shape=input_shape, # <-- TF bug here! (None shape is not translated from None input_shape) 
                #shape=(None, input_shape[1], input_shape[2], input_shape[3]),
                shape=input_shape,
                trainable=False, use_resource=True, dtype=tf.float32)

    
    def get_random_hole_widths_heights(self, bsz, hole_minw, hole_maxw, 
                                       hole_minh, hole_maxh):
        """ Randomize hole width and heights. """
        if hole_minw==hole_maxw:
            xs_w = hole_minw
        else:
            xs_w = tf.random.uniform((bsz,self.n_holes), hole_minw, hole_maxw,
                tf.int32) # (b,n): width of x for bth batch in nth hole
            
        if hole_minh==hole_maxh:
            ys_h = hole_minh
        else:
            ys_h = tf.random.uniform((bsz,self.n_holes), hole_minh, hole_maxh,
                tf.int32) # (b,n): height of y for bth batch in nth hole
        return xs_w, ys_h
    
        
    def get_random_hole_positions(self, bsz, x_w, x_h):
        """ Randomize hole positions (center pos of x and y). """
        xs = tf.random.uniform((bsz, self.n_holes), 0, x_w - 1, tf.int32)
            # (b,n): pos center x of (batch, n_hole) 
        ys = tf.random.uniform((bsz, self.n_holes), 0, x_h - 1, tf.int32)
            # (b,n): pos center y of (batch, n_hole) 
        return xs, ys 


    @tf.function
    def call(self, x):
        if self.prob > 0:
            bsz = tf.shape(x)[0]
            x_h = tf.shape(x)[1]
            x_w = tf.shape(x)[2]
            
            self.varx.assign(x)

            if self.hole_minw==None:
                hole_minw = (x_w // 10)
            elif self.hole_minw==-1:
                hole_minw = x_w - 1
            else:
                hole_minw = self.hole_minw
            
            if self.hole_maxw==None:
                hole_maxw = tf.cast((tf.cast(x_w, tf.float32) / 2.5), tf.int32)
            elif self.hole_maxw==-1:
                hole_maxw = x_w - 1
            else:
                hole_maxw = self.hole_maxw
                
            if self.hole_minh==None:
                hole_minh = (x_h // 10)
            elif self.hole_minh==-1:
                hole_minh = x_h - 1
            else:
                hole_minh = self.hole_minh
            
            if self.hole_maxh==None:
                hole_maxh = tf.cast((tf.cast(x_h, tf.float32) / 2.5), tf.int32)
            elif self.hole_maxh==-1:
                hole_maxh = x_h - 1
            else:
                hole_maxh = self.hole_maxh
            
            # Randomize hole width and heights
            if self.uniform_mask:
                xs_w, ys_h = self.get_random_hole_widths_heights(
                    1, hole_minw, hole_maxw, hole_minh, hole_maxh)
                xs_w = tf.reshape(xs_w, (-1,))
                ys_h = tf.reshape(ys_h, (-1,))   
            else:
                xs_w, ys_h = self.get_random_hole_widths_heights(
                    bsz, hole_minw, hole_maxw, hole_minh, hole_maxh)
            
            # Randomize hole positions                
            if self.uniform_mask:
                xs, ys = self.get_random_hole_positions(1, x_w, x_h)
                xs = tf.reshape(xs, (-1,))
                ys = tf.reshape(ys, (-1,)) 
            else:
                xs, ys = self.get_random_hole_positions(bsz,x_w, x_h)
            
            
            # Center positions for horizontal/vertical mode
            if (self.hole_minw==-1) & (self.hole_maxw==-1): # 'horizontal' mode
                if self.uniform_mask:
                    xs = tf.ones(self.n_holes, tf.int32) * (x_w // 2)
                else:
                    xs = tf.ones((bsz, self.n_holes), tf.int32) * (x_w // 2)
                    # xs: center position on x-axis for (batch, n_hole)
            
            if (self.hole_minh==-1) & (self.hole_maxh==-1): # 'vertical' mode
                if self.uniform_mask:
                    ys = tf.ones(self.n_holes, tf.int32) * (x_h // 2)
                else:
                    ys = tf.ones((bsz, self.n_holes), tf.int32) * (x_h // 2)
                    # ys: center position on y-axis for (batch, n_hole)
            
            # Hole ranges
            xs_start = tf.clip_by_value(xs - (xs_w // 2), 0, x_w - 2)
            xs_end = tf.clip_by_value(xs + (xs_w // 2), 1, x_w - 1)
            ys_start = tf.clip_by_value(ys - (ys_h // 2), 0, x_h - 2)
            ys_end = tf.clip_by_value(ys + (ys_h // 2), 1, x_h - 1)
            
            # Final hole lengths (this can be different from xs_w and ys_h)
            ws = xs_end - xs_start
            hs = ys_end - ys_start
            
            # Activation of augmentation holes with given prob.
            if self.uniform_mask:
                #hole_act = tf.random.uniform([self.n_holes], 0., 1.)
                hole_act = tf.random.uniform([bsz], 0., 1.)
                hole_act = tf.less(hole_act, self.prob) # dtype: tf.bool, shape: N
            else:
                hole_act = tf.random.uniform([bsz, self.n_holes], 0., 1.)
                hole_act = tf.less(hole_act, self.prob) # dtype: tf.bool, shape: BxN

            
            # Generate Holes
            if self.uniform_mask:
                for b in tf.range(bsz):
                    if hole_act[b]:
                        for n in tf.range(self.n_holes):
                            if self.hole_fill=='min':
                                hole_area = tf.zeros([hs[n], ws[n], 1]) + tf.reduce_min(x[b,:,:,:])
                            elif self.hole_fill=='zeros':
                                hole_area = tf.zeros([hs[n], ws[n], 1])
                            elif self.hole_fill=='random':
                                hole_area = tf.random.uniform([hs[n], ws[n], 1],
                                                              tf.reduce_min(x[b,:,:,:]),
                                                              tf.reduce_max(x[b,:,:,:]),
                                                              tf.float32)
                            elif self.hole_fill=='random_with_range':
                                hole_area = tf.random.uniform([hs[n], ws[n], 1],
                                                              self.filler_min,
                                                              self.filler_max,
                                                              tf.float32)
                            else:
                                raise NotImplementedError(self.hole_fill)    
                            self.varx[b,ys_start[n]:ys_end[n],xs_start[n]:xs_end[n],:].assign(hole_area)
                    else:
                        pass; # bypass this patch
            else:
                for b in tf.range(bsz):
                    for n in tf.range(self.n_holes):
                        if hole_act[b,n]:
                            if self.hole_fill=='min':
                                hole_area = tf.zeros([hs[b,n], ws[b,n], 1]) + tf.reduce_min(x[b,:,:,:])
                            elif self.hole_fill=='zeros':
                                hole_area = tf.zeros([hs[b,n], ws[b,n], 1])
                            elif self.hole_fill=='random':
                                hole_area = tf.random.uniform([hs[b,n], ws[b,n], 1],
                                                              tf.reduce_min(x[b,:,:,:]),
                                                              tf.reduce_max(x[b,:,:,:]),
                                                              tf.float32)
                            elif self.hole_fill=='random_with_range':
                                hole_area = tf.random.uniform([hs[b,n], ws[b,n], 1],
                                                              self.filler_min,
                                                              self.filler_max,
                                                              tf.float32)
                            else:
                                raise NotImplementedError(self.hole_fill)    
                            self.varx[b,ys_start[b,n]:ys_end[b,n],xs_start[b,n]:xs_end[b,n],:].assign(hole_area)
                        else:
                            pass; # bypass this patch
                            
            return self.varx.value()
        else:
            # bypass...
            return x

    
    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, input_shape[1], input_shape[2], input_shape[3]])


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'prob': self.prob,
            'n_holes': self.n_holes,
            'uniform_mask': self.uniform_mask,
            'hole_fill': self.hole_fill,
            'filler_min': self.filler_min,
            'filler_max': self.filler_max,
            'hole_minw': self.hole_minw,
            'hole_maxw': self.hole_maxw,
            'hole_minh': self.hole_minh,
            'hole_maxh': self.hole_maxh,
            'varx': np.zeros(self.varx.shape)
        })
        return config
    

def plot_to_image(figure):
    """ Converts the matplotlib figure to a PNG. """
    # Save the plot to a PNG in memory.
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Add batch dimension
    image = tf.expand_dims(image, 0)
    return image


def display_spec(mel_spectrogram=None, title=None, get_img=None):
    """

    Arguments
    ---------
      mel_spectrogram: (ndarray)
          mel_spectrogram to visualize.
      title: (String)
          plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    fig = plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, origin='lower')
    plt.xlabel('time(frame)')
    plt.ylabel('mel')
    if title: plt.title(title)
    plt.tight_layout()
    if get_img:
        return plot_to_image(fig)
        print('GET_IMSHOW: created an image for tensorboard...')
    else:
        plt.show()
    return


def test():
    import librosa
    # Load audio file, extract mel spectrogram
    audio, sampling_rate = librosa.load('data/61-70968-0002.wav')
    m = librosa.feature.melspectrogram(y=audio,
                                       sr=sampling_rate,
                                       n_mels=256,
                                       hop_length=128,
                                       fmax=8000)
    m = librosa.power_to_db(abs(m))

    # Reshape spectrogram shape to [batch_size, time, frequency, 1]
    m = tf.constant(np.reshape(m, (1, m.shape[0], m.shape[1], 1)))
    # Show original spec
    display_spec(m[0,:,:,0].numpy(), title='Original')
    
    # Apply Cutout
    spec_ncutout_layer = SpecNCutout(prob=0.5,
                                     n_holes=3,
                                     uniform_mask=False,
                                     hole_fill='random')
    spec_ncutout_layer.trainable = False
    m_aug = spec_ncutout_layer(m) 
    display_spec(m_aug[0,:,:,0].numpy())
    
    # Test speed with large batch input.
    m_batch = tf.stack([m[0,:,:,:]] * 128) # batch size is 1024
    
    spec_ncutout_layer = SpecNCutout(prob=0.5,
                                 n_holes=3,
                                 uniform_mask=False,
                                 hole_fill='random')
    spec_ncutout_layer.trainable = False
    m_batch_aug = spec_ncutout_layer(m_batch)
    display_spec(m_batch_aug[14,:,:,0].numpy())
    #%timeit -n 10 spec_ncutout_layer(m_batch)
    #%timeit -n 10 spec_ncutout_layer(spec_ncutout_layer(m_batch))
    # Uniform=True, 3holes:
    #    3.13 ms ± 158 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    # Uniform=False, 3holes:
    #    205 ms ± 2.83 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)   

