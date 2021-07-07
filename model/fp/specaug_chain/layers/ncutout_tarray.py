# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" ncutout_tarray.py

SpecNCutout: Batch-wise SpecAugment + N-CutOut layers for augmentation in 
spectral domain for GPUs.

• variable number of holes for cutout
• hole filler types ['zeros', 'random']
• compatiable with TF2.x and @tf.function decorator
• implementation based on tf.TensorArray


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
    SpecNCutout based on SpecAugment and CutOut: implementation using
    tf.TensorArray.
    
    Arguments
    ---------        
    • prob: (tf.Float)
        probability (0-1) of cutout activity. If prob=1.0 (default), always
        output cut-out spectrogram
    • n_holes: (int)
        Number of random holes to create
    • uniform_mask: (bool)
        If True (default), use efficient uniform mask in batch. 
    • hole_fill: (str) or [(float), (float)]
        Values to fill the holes with. Default is 'min'.
        - 'min' fills with minimum magnitude of input spectrogram.
        - 'zeros' fills with zeros.
        - 'random' fills with random values within the range of min and max of input.
        - [min_mag, max_mag] fills with random values within the range.
    • hole_config: [(int),(int),(int),(int)]
        Configures the range of hole mask size by [min_width, max_width,
        min_height, max_height]. If [None,None,None,None] (default), 1/10 of
        the input length will be set as minimum, and 1/2 of the input length
        will be set as maximum.
    • hole_wise_act_with_prob: (bool)
        - if True (default), apply hole-wise activation probability to
          augmentation.  
        - if False, element-wise activation is uniformly applied to all the
          corresponding holes. 
            
    Input
    -----
    • <tf.Float> 4D tensor variable with shape (B,H,W,C), equivalent with
      (Batch,Freq,Time,1).
    
    """
    def __init__(self,
                 prob=1.0,
                 n_holes=1,
                 uniform_mask=True,
                 hole_fill='min',
                 hole_config=[None,None,None,None],
                 hole_wise_act_with_prob=True,
                 **kwargs):
        super(SpecNCutout, self).__init__()
        self.prob = prob
        self.n_holes = n_holes
        self.uniform_mask = uniform_mask
        self.hole_fill, self.filler_min, self.filler_max = None, None, None
        if hole_fill in ['min', 'zeros', 'random']:
            self.hole_fill = hole_fill
        elif type(hole_fill)==list and len(hole_fill)==2:
            self.hole_fill = 'random_with_range'
            self.filler_min = hole_fill[0]
            self.filler_max = hole_fill[1]
        else:
            raise NotImplementedError(hole_fill)
            
        self.hole_minw, self.hole_maxw, self.hole_minh, self.hole_maxh = hole_config
     

    def build(self, input_shape):
        self.x_h = input_shape[1]
        self.x_w = input_shape[2]
        self.index_h = tf.range(self.x_h) # [0,1,2,...h]: will be used for 1d-mask
        self.index_w = tf.range(self.x_w) # [0,1,2,...w]: will be used for 1d-mask
        
        # Pre-calculate hole fillers
        if self.hole_fill=='min':
            self.hf = tf.ones(input_shape, tf.float32)
        elif self.hole_fill=='zeros':
            self.hf = tf.zeros(input_shape, tf.float32)
        elif self.hole_fill=='random':
            self.hf = tf.random.uniform(input_shape, 0., 1., tf.float32)
        elif self.hole_fill=='random_with_range':
            self.hf = tf.random.uniform(input_shape, self.filler_min,
                                   self.filler_max, tf.float32)        

    
    def generate_single_mask(self, h_start=int(), h_end=int(), w_start=int(),
                             w_end=int()):
        # 1d mask
        m_h = tf.logical_and((h_start <= self.index_h), (self.index_h <= h_end))  
        m_w = tf.logical_and((w_start <= self.index_w), (self.index_w <= w_end))
        
        # from 1d to 2d mask
        m_h = tf.expand_dims(m_h, axis=1) # shape: (h,1)
        m_w = tf.expand_dims(m_w, axis=0) # shape: (1,w)
        mask = tf.logical_and(m_h, m_w) # shape:(h,w), dtype:tf.bool
        return mask # (h, w)
    
    
    def generate_mixed_mask(self, bsz, hole_act_prob, hole_minh, hole_maxh,
                            hole_minw, hole_maxw):
        # If using uniform mask, b should be set to 1.
        # Else if using different masks for batches, b should be > 1.
        n_masks = self.n_holes
        
        # Randomize hole width and heights
        if hole_minw==hole_maxw:
            xs_w = hole_minw
        else:
            xs_w = tf.random.uniform((bsz, n_masks), hole_minw, hole_maxw, tf.int32)
            # (n): width of x for bth batch in nth mask
        
        if hole_minh==hole_maxh:
            ys_h = hole_minh
        else:
            ys_h = tf.random.uniform((bsz, n_masks), hole_minh, hole_maxh, tf.int32)
                # (n): height of y for bth batch in nth mask
        
        # Get hole positions
        if (self.hole_minw==-1) & (self.hole_maxw==-1): # 'horizontal' mode
            xs = tf.ones((bsz, n_masks), tf.int32) * (self.x_w // 2)
        else:
            # Randomize hole positions on x-axis
            xs = tf.random.uniform((bsz, n_masks), 0, self.x_w - 1, tf.int32) 
                # xs: center position on x-axis for (batch, n_hole)
                
        if (self.hole_minh==-1) & (self.hole_maxh==-1): # 'vertical' mode    
            ys = tf.ones((bsz, n_masks), tf.int32) * (self.x_h // 2)
        else:
            ys = tf.random.uniform((bsz, n_masks), 0, self.x_h - 1, tf.int32) 
                # ys: center position on y-axis for (batch, n_hole)
       
        # Hole ranges
        xs_start = tf.clip_by_value(xs - (xs_w // 2), 0, self.x_w - 2)
        xs_end = tf.clip_by_value(xs + (xs_w // 2), 1, self.x_w - 1)
        ys_start = tf.clip_by_value(ys - (ys_h // 2), 0, self.x_h - 2)
        ys_end = tf.clip_by_value(ys + (ys_h // 2), 1, self.x_h - 1)
        
        # Generate single mask, n times --> mix masks
        m_arr =  tf.TensorArray(tf.bool, size=bsz, dynamic_size=False,
                                clear_after_read=True)
        for b in tf.range(bsz):
            mask = tf.zeros((self.x_h, self.x_w), tf.bool)
            for n in tf.range(self.n_holes):
                if tf.random.uniform([]) < hole_act_prob:
                    _mask = self.generate_single_mask(ys_start[b,n], ys_end[b,n],
                                                      xs_start[b,n], xs_end[b,n])
                    _mask.set_shape(mask.get_shape())
                    # Mixing
                    mask = tf.logical_or(mask, _mask) # shape: (h,w)
            mask = tf.expand_dims(mask, axis=2) # shape: (h,w,1)
            # Stack batch of masks (if bsz>0)
            m_arr = m_arr.write(b, mask) # shape: (b,h,w,1) with True for hole area 
            
        return m_arr.stack() # shape: (b,h,w,1) with True for HOLE area (background is False)
        
    
    @staticmethod
    def get_true_background(mask):
        return tf.cast(tf.logical_not(mask), tf.float32)
    

    
    @staticmethod
    def get_true_holes(mask):
        return tf.cast(mask, tf.float32)

                  
    def get_hole_filler(self, x):
        """ Generate hole filler that has same shape with input spectrogram. """
        hf = self.hf
        if self.hole_fill=='min':
            hf = hf * tf.reduce_mean(x)
        elif self.hole_fill=='zeros':
            hf = hf
        elif self.hole_fill=='random':
            hf = hf * (tf.reduce_max(x) - tf.reduce_min(x)) + tf.reduce_min(x)
        elif self.hole_fill=='random_with_range':
            hf = hf 
        return hf
    
    
    @tf.function
    def call(self, x):
        # if tf.random.uniform([], 0, 1) < self.prob: 
        if self.prob > 0:
            bsz = tf.shape(x)[0]
            x_h = tf.shape(x)[1]
            x_w = tf.shape(x)[2]

            if self.hole_minw==None:
                hole_minw = (x_w // 10)
            elif self.hole_minw==-1:
                hole_minw = x_w
            else:
                hole_minw = self.hole_minw
            
            if self.hole_maxw==None:
                hole_maxw = tf.cast((tf.cast(x_w, tf.float32) / 2.5), tf.int32)
            elif self.hole_maxw==-1:
                hole_maxw = x_w
            else:
                hole_maxw = self.hole_maxw
                
            if self.hole_minh==None:
                hole_minh = (x_h // 10)
            elif self.hole_minh==-1:
                hole_minh = x_h
            else:
                hole_minh = self.hole_minh
            
            if self.hole_maxh==None:
                hole_maxh = tf.cast((tf.cast(x_h, tf.float32) / 2.5), tf.int32)
            elif self.hole_maxh==-1:
                hole_maxh = x_h
            else:
                hole_maxh = self.hole_maxh
    
            
            # Generation of masks
            if self.uniform_mask:
                # Uniform mask: set bsz=1 for uniform mask generation
                uni_mask = self.generate_mixed_mask(1, 1., hole_minh,
                                                    hole_maxh, hole_minw,
                                                    hole_maxw) # shape: (b,h,w,1)
                
                # Random activation within batch
                act_mask = tf.cast((tf.random.uniform([bsz, 1, 1, 1], 0, 1) < self.prob), tf.float32)
                
                # Apply mask
                x_org = x * (1 - act_mask) # keep original without augmentation
                x_aug = x * act_mask
                x_aug = x_aug * self.get_true_background(uni_mask) # * act_mask
                x_aug = x_aug + self.get_true_holes(uni_mask) * act_mask * self.get_hole_filler(x)
                
                # Merge
                x = x_org + x_aug
            else:
                # Generate different masks for elements with independent prob
                masks = self.generate_mixed_mask(bsz, self.prob, hole_minh,
                                                 hole_maxh, hole_minw,
                                                 hole_maxw)
                
                # Apply different masks
                x = x * self.get_true_background(masks) + self.get_true_holes(masks) * self.get_hole_filler(x)
            return x
        else:
            # bypass...
            return x


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'x_h': self.x_h,
            'x_w': self.x_w,
            'index_h': np.arange(self.x_h),
            'index_w': np.arange(self.x_w),
            'prob': self.prob,
            'n_holes': self.n_holes,
            'hole_fill': self.hole_fill,
            'filler_min': self.filler_min,
            'filler_max': self.filler_max,
            'hole_minw': self.hole_minw,
            'hole_maxw': self.hole_maxw,
            'hole_minh': self.hole_minh,
            'hole_maxh': self.hole_maxh,
            'uniform_mask': self.uniform_mask
        })
        return config
    
    
    
#%% TEST   
def plot_to_image(figure):
    """ Converts the matplotlib figure to a PNG. """
    # Save the plot to a PNG in memory.
    buf = BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
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
    spec_ncutout_layer = SpecNCutout(prob=1.,
                                     n_holes=1,
                                     uniform_mask=False,
                                     hole_fill='min',
                                     hole_wise_act_with_prob=True,
                                     hole_config=[-1,-1,None,None])
    spec_ncutout_layer.trainable = False
    m_aug = spec_ncutout_layer(m) 
    display_spec(m_aug[0,:,:,0].numpy())
    
    # Test speed with large batch input.
    m_batch = tf.stack([m[0,:,:,:]] * 128) # batch size is 1024
    m_batch_aug = spec_ncutout_layer(m_batch)
    display_spec(m_batch_aug[1,:,:,0].numpy())
    #%timeit -n 10 spec_ncutout_layer(m_batch)
    #%timeit -n 10 spec_ncutout_layer(spec_ncutout_layer(m_batch))

