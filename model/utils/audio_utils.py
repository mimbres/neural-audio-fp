# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" audio_utils.py """
import wave
import numpy as np


def max_normalize(x):
    """
    Parameters
    ----------
    x : (float)

    Returns
    -------
    (float)
        Max-normalized audio signal.

    """
    if np.max(np.abs(x)) == 0:
        return x
    else:
        return x / np.max(np.abs(x))


def background_mix(x, x_bg, fs, snr_db):
    """
    Parameters
    ----------
    x : 1D array (float)
        Input audio signal.
    x_bg : 1D array (float)
        Background noise signal.
    fs : (float)
        Sampling rate.
    snr_db : (float)
        signal-to-noise ratio in decibel.

    Returns
    -------
    1D array
        Max-normalized mix of x and x_bg with SNR

    """
    # Check length
    if len(x) > len(x_bg):  # This will not happen though...
        _x_bg = np.zeros(len(x))
        bg_start = np.random.randint(len(x) - len(x_bg))
        bg_end = bg_start + len(x_bg)
        _x_bg[bg_start:bg_end] = x_bg
        x_bg = _x_bg
    elif len(x) < len(x_bg):  # This will not happen though...
        bg_start = np.random.randint(len(x_bg) - len(x))
        bg_end = bg_start + len(x)
        x_bg = x_bg[bg_start:bg_end]
    else:
        pass

    # Normalize with energy
    rmse_bg = np.sqrt(np.sum(x_bg**2 / len(x_bg)))
    x_bg = x_bg / rmse_bg
    rmse_x = np.sqrt(np.sum(x**2) / len(x))
    x = x / rmse_x

    # Mix
    magnitude = np.power(10, snr_db / 20.)
    x_mix = magnitude * x + x_bg
    return max_normalize(x_mix)


def log_scale_random_number_batch(bsz=int(), amp_range=(0.1, 1.)):
    range_log = np.log10(amp_range)
    random_number_log = np.random.rand(bsz) * (
        np.max(range_log) - np.min(range_log)) + np.min(range_log)
    return np.power(10, random_number_log)


def bg_mix_batch(event_batch,
                 bg_batch,
                 fs,
                 snr_range=(6, 24),
                 unit='db',
                 mode='energy'):
    X_bg_mix = np.zeros((event_batch.shape[0], event_batch.shape[1]))

    # Random SNR
    min_snr = np.min(snr_range)
    max_snr = np.max(snr_range)
    snrs = np.random.rand(len(event_batch))
    snrs = snrs * (max_snr - min_snr) + min_snr

    # Random amp (batch-wise)
    event_amp_ratio_batch = log_scale_random_number_batch(bsz=len(event_batch),
                                                          amp_range=(0.1, 1))

    for i in range(len(event_batch)):
        event_max = np.max(np.abs(event_batch[i]))
        bg_max = np.max(np.abs(bg_batch[i]))
        #event_amp_ratio = log_scale_random_number(amp_range=(0.01,1))

        if event_max == 0 or bg_max == 0:
            X_bg_mix[i] = event_batch[i] + bg_batch[i]
            X_bg_mix[i] = max_normalize(X_bg_mix[i])

        else:
            X_bg_mix[i] = background_mix(x=event_batch[i],
                                         x_bg=bg_batch[i],
                                         fs=fs,
                                         snr_db=snrs[i])
        X_bg_mix[i] = event_amp_ratio_batch[i] * X_bg_mix[i]

    return X_bg_mix


def ir_aug_batch(event_batch, ir_batch):
    n_batch = len(event_batch)
    X_ir_aug = np.zeros((n_batch, event_batch.shape[1]))

    for i in range(n_batch):
        x = event_batch[i]
        x_ir = ir_batch[i]

        # FFT -> multiply -> IFFT
        fftLength = np.maximum(len(x), len(x_ir))
        X = np.fft.fft(x, n=fftLength)
        X_ir = np.fft.fft(x_ir, n=fftLength)
        x_aug = np.fft.ifft(np.multiply(X_ir, X))[0:len(x)].real
        if np.max(np.abs(x_aug)) == 0:
            pass
        else:
            x_aug = x_aug / np.max(np.abs(x_aug))  # Max-normalize

        X_ir_aug[i] = x_aug

    return X_ir_aug


def get_fns_seg_list(fns_list=[],
                     segment_mode='all',
                     fs=22050,
                     duration=1,
                     hop=None):
    """
    return: fns_event_seg_list
        
        [[filename, seg_idx, offset_min, offset_max], [ ... ] , ... [ ... ]]
        
        offset_min is 0 or negative integer
        offset_max is 0 or positive integer
        
    """
    if hop == None: hop = duration
    fns_event_seg_list = []

    for offset_idx, filename in enumerate(fns_list):
        # Get audio info
        n_frames_in_seg = fs * duration
        n_frames_in_hop = fs * hop  # 2019 09.05
        file_ext = filename[-3:]

        if file_ext == 'wav':
            pt_wav = wave.open(filename, 'r')
            _fs = pt_wav.getframerate()

            if fs != _fs:
                raise ValueError('Sample rate should be {} but got {}'.format(
                    str(fs), str(_fs)))

            n_frames = pt_wav.getnframes()
            #n_segs = n_frames // n_frames_in_seg
            if n_frames > n_frames_in_seg:
                n_segs = (n_frames - n_frames_in_seg +
                          n_frames_in_hop) // n_frames_in_hop
            else:
                n_segs = 1

            n_segs = int(n_segs)
            assert (n_segs > 0)
            residual_frames = np.max([
                0,
                n_frames - ((n_segs - 1) * n_frames_in_hop + n_frames_in_seg)
            ])
            pt_wav.close()
        else:
            raise NotImplementedError(file_ext)

        # 'all', 'random_oneshot', 'first'
        if segment_mode == 'all':
            for seg_idx in range(n_segs):
                offset_min, offset_max = int(-1 *
                                             n_frames_in_hop), n_frames_in_hop
                if seg_idx == 0:  # first seg
                    offset_min = 0
                if seg_idx == (n_segs - 1):  # last seg
                    offset_max = residual_frames

                fns_event_seg_list.append(
                    [filename, seg_idx, offset_min, offset_max])
        elif segment_mode == 'random_oneshot':
            seg_idx = np.random.randint(0, n_segs)
            offset_min, offset_max = n_frames_in_hop, n_frames_in_hop
            if seg_idx == 0:  # first seg
                offset_min = 0
            if seg_idx == (n_segs - 1):  # last seg
                offset_max = residual_frames
            fns_event_seg_list.append(
                [filename, seg_idx, offset_min, offset_max])
        elif segment_mode == 'first':
            seg_idx = 0
            offset_min, offset_max = 0, 0
            fns_event_seg_list.append(
                [filename, seg_idx, offset_min, offset_max])
        else:
            raise NotImplementedError(segment_mode)

    return fns_event_seg_list


def load_audio(filename=str(),
               seg_start_sec=float(),
               offset_sec=0.0,
               seg_length_sec=float(),
               seg_pad_offset_sec=0.0,
               fs=22050,
               amp_mode='normal'):
    """
        Open file to get file info --> Calulate index range
        --> Load sample by index --> Padding --> Max-Normalize --> Out
        
    """
    start_frame_idx = np.floor((seg_start_sec + offset_sec) * fs).astype(int)
    seg_length_frame = np.floor(seg_length_sec * fs).astype(int)
    end_frame_idx = start_frame_idx + seg_length_frame

    # Get file-info
    file_ext = filename[-3:]
    #print(start_frame_idx, end_frame_idx)

    if file_ext == 'wav':
        pt_wav = wave.open(filename, 'r')
        pt_wav.setpos(start_frame_idx)
        x = pt_wav.readframes(end_frame_idx - start_frame_idx)
        x = np.frombuffer(x, dtype=np.int16)
        x = x / 2**15  # dtype=float
    else:
        raise NotImplementedError(file_ext)

    # Max Normalize, random amplitude
    if amp_mode == 'normal':
        pass
    elif amp_mode == 'max_normalize':
        _x_max = np.max(np.abs(x))
        if _x_max != 0:
            x = x / _x_max
    else:
        raise ValueError('amp_mode={}'.format(amp_mode))

    # padding process. it works only when win_size> audio_size and padding='random'
    audio_arr = np.zeros(int(seg_length_sec * fs))
    seg_pad_offset_idx = int(seg_pad_offset_sec * fs)
    audio_arr[seg_pad_offset_idx:seg_pad_offset_idx + len(x)] = x
    return audio_arr


def load_audio_multi_start(filename=str(),
                           seg_start_sec_list=[],
                           seg_length_sec=float(),
                           fs=22050,
                           amp_mode='normal'):
    """ Load_audio wrapper for loading audio with multiple start indices. """
    # assert(len(seg_start_sec_list)==len(seg_length_sec))
    out = None
    for seg_start_sec in seg_start_sec_list:
        x = load_audio(filename=filename,
                       seg_start_sec=seg_start_sec,
                       seg_length_sec=seg_length_sec,
                       fs=8000)
        x = x.reshape((1, -1))
        if out is None:
            out = x
        else:
            out = np.vstack((out, x))
    return out  # (B,T)


def npy_to_wav(root_dir=str(), source_fs=int(), target_fs=int()):
    import wavio, glob, scipy
    import numpy as np

    fns = glob.glob(root_dir + '**/*.npy', recursive=True)
    for fname in fns:
        audio = np.load(fname)
        resampled_length = int(len(audio) / source_fs * target_fs)
        audio = scipy.signal.resample(audio, resampled_length)
        audio = audio * 2**15
        audio = audio.astype(np.int16)  # 16-bit PCM
        wavio.write(fname[:-4] + '.wav', audio, target_fs, sampwidth=2)
