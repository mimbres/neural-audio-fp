# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" specaug_chain.py

Spec-augmentation Chainer: build a custom chain of batch-wise bugmentation for
spectogram with GPUs.

    • SpecAugment + N-CutOut
    • supports GPUs
    • supports variable number of holes for cutout
    • supports hole filler types ['zeros', 'random']
    • compatiable with TF2.x and @tf.function decorator


USAGE:
    
    spec_ncutout_layer = SpecNCutout(prob=0.5,
                                     n_holes=3,
                                     hole_fill='random')
    m = (your method to get spectrogram here...)
    m_aug = spec_ncutout_layer(m) 
        
For more details, see test_chain() in the below.

References:
    • Original numpy implementation of https://arxiv.org/abs/1708.04552
        (no N, no GPU, no batch)
        https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    • Random erasing example code
        https://github.com/uranusx86/Random-Erasing-tensorflow

"""
import tensorflow as tf
import numpy as np              # for test()
import matplotlib.pyplot as plt # for test()
from io import BytesIO          # for test()
#from model.fp.specaug_chain.layers.ncutout_var import SpecNCutout
from model.fp.specaug_chain.layers.ncutout_tarray import SpecNCutout


class SpecAugChainer(tf.keras.layers.Layer):
    """ SpecAugChainer
       
    Arguments
    ---------    
    (General)
    
    • chain_config: (list)(str) design an ordered chain of augmentations.
                 ex) ['cutout', 'vertical', 'horizontal'] 
        
    • probs: (float32) probability (0-1) of cutout activity. Default is 0.5
                 If prob=1.0, apply augmentation to all the outputs in batch.
             (list)(float32) a list of probabilitiy, ex) [0.5, 0.5, 0.7]
    
    • uniform_mask: If True (default), use efficient uniform mask in batch. 
    
    (Cutout parameters)
    • n_holes: number of random holes to create. Default is 1.
            If 'cutout' is not in chain_config, ignore this parameter.
    
    • hole_fill: (str) or [(float), (float)]
            Values to fill the holes with. 'min'(default), 'zeros', 'random',
                or [min_mag, max_mag]
            - 'min' fills with minimum magnitude of input spectrogram.
            - 'zeros' fills with zeros.
            - 'random' fills with random values within range of min and max of input.
            - [min_mag, max_mag] fills with random values within the given range.
    
    • hole_config: [(int),(int),(int),(int)] Configuring the range of hole
                size as [min_width, max_width, min_height, max_height]
                If [None,None,None,None] (default), 1/10 of input length is 
                set as minimum, and 1/2 of input length is set as maximum.


    Input:
    - <tf.tensor> 4D tensor variable with shape (B,H,W,C), equivalent with
                  (Batch,Freq,Time,1).

    
    Activity control:
    - bypass: (bool) You can disable/enable augmentation by setting
            self.bypass=True/False in your training/test loop.
                  
    """
    def __init__(self,
                 chain_config=['cutout'],
                 probs=1.0, # [1.0, 0.5]
                 uniform_mask=True,
                 n_holes=1,
                 hole_fill='min',
                 hole_config=[None,None,None,None],
                 **kwargs):
        super(SpecAugChainer, self).__init__()
        
        self.chain_config = chain_config
        self.probs = probs if type(probs)==list else [probs]
        if len(self.probs) < len(self.chain_config):
            self.probs = self.probs * len(self.chain_config) # [1] --> [1,1,1]
        self.uniform_mask = uniform_mask
        self.hole_fill = hole_fill
        if 'cutout' in self.chain_config:
            self.n_holes = n_holes
            self.hole_config = hole_config
        else:
            self.n_holes = 'DISABLED (only interacts with cutout)'
            self.hole_config = 'DISABLED (only interacts with cutout)'
        self.bypass = False

        # Create a chain
        self.chain = tf.keras.Sequential()
        for i, l_name in enumerate(self.chain_config):
            if l_name=='cutout':
                self.chain.add(
                    SpecNCutout(
                        name=f'{str(i)}_{l_name}',
                        prob=self.probs[i],
                        n_holes=self.n_holes,
                        uniform_mask=self.uniform_mask,
                        hole_fill=self.hole_fill,
                        hole_config=self.hole_config,
                        trainable=False))
            elif l_name=='vertical':
                self.chain.add(
                    SpecNCutout(
                        name=f'{str(i)}_{l_name}',
                        prob=self.probs[i],
                        n_holes=1,
                        uniform_mask=self.uniform_mask,
                        hole_fill=self.hole_fill,
                        hole_config=[5, 16 , -1, -1]))
                        # hole_config=[None, None, -1, -1]))
            elif l_name=='horizontal':
                self.chain.add(
                    SpecNCutout(
                        name=f'{str(i)}_{l_name}',
                        prob=self.probs[i],
                        n_holes=1,
                        uniform_mask=self.uniform_mask,
                        hole_fill=self.hole_fill,
                        hole_config=[-1, -1, 5, 20]))
                        # hole_config=[-1, -1, None, None]))
            else:
                raise NotImplementedError(l_name)
    
    
    @tf.function
    def call(self, x):
        if self.bypass:
            return x
        else:
            return self.chain(x)
    
    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, input_shape[1], input_shape[2], input_shape[3]])
    
    
    def get_config(self): # this is requirements to tf.keras.Model.save()
        config = super().get_config().copy()
        config.update({
            'chain_config': self.chain_config,
            'probs': self.probs,
            'uniform_mask': self.uniform_mask,
            'hole_fill': self.hole_fill,
            'n_holes': self.n_holes,
            'hole_config': self.hole_config,
            'chain': self.chain
        })
        return config

    
def get_specaug_chain_layer(cfg, trainable=False):
    """
    Get spec-augment chain layer from neural audio fingerprinter config file.

    Parameters
    ----------
    cfg : (dict)
        created from the '.yaml' located in /config dicrectory

    Returns
    -------
    <SpecAugChainer> object

    """
    m = SpecAugChainer(chain_config=cfg['SPEC_AUG']['SPECAUG_CHAIN'],
                       probs=cfg['SPEC_AUG']['SPECAUG_PROBS'],
                       n_holes=cfg['SPEC_AUG']['SPECAUG_N_HOLES'],
                       hole_fill=cfg['SPEC_AUG']['SPECAUG_HOLE_FILL'])
    m.trainable = trainable
    return m
    

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    
    """
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
    Visualizing a sample output of specaug-chain for Tensorboard

    Arguments
    ---------
    mel_spectrogram: (ndarray)
        Input (mel-)spectrogram to visualize.
    title: (String)
        Title of the figure.
     
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


def test_chain():
    # Build chain
    input_mel = tf.keras.layers.Input(batch_size=64, shape=(128, 96, 1), name='input_mel')
    augmented_mel = SpecAugChainer(chain_config=['cutout', 'vertical', 'horizontal'],
                                   probs=1.0, uniform_mask=True,
                                   n_holes=3, hole_fill='random')(input_mel)
    m_aug = tf.keras.Model(inputs=[input_mel], outputs=[augmented_mel], name='m_aug')
    m_aug.build(input_mel.get_shape())
    m_aug.trainable = False
    m_aug.summary()
    
    # Generate data
    import librosa
    # Load audio file, extract mel spectrogram
    audio, sampling_rate = librosa.load('data/61-70968-0002.wav')
    d = librosa.feature.melspectrogram(y=audio,
                                       sr=sampling_rate,
                                       n_mels=128,
                                       hop_length=128,
                                       fmax=8000)
    d = d[:, :96]
    d = librosa.power_to_db(abs(d))

    # Reshape spectrogram shape to [batch_size, time, frequency, 1]
    d = tf.constant(np.reshape(d, (1, d.shape[0], d.shape[1], 1)))
    # Show original spec
    display_spec(d[0,:,:,0].numpy(), title='Original')
    
    d_batch = tf.stack([d[0,:,:,:]] * 64) # batch size is 64
    d_batch_aug = m_aug(d_batch)
    display_spec(d_batch_aug[42,:,:,0].numpy())
