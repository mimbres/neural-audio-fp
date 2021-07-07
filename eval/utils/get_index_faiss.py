# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import faiss
import numpy as np


def get_index(index_type,
              train_data,
              train_data_shape,
              use_gpu=True,
              max_nitem_train=2e7):
    """
    • Create FAISS index
    • Train index using (partial) data
    • Return index

    Parameters
    ----------
    index_type : (str)
        Index type must be one of {'L2', 'IVF', 'IVFPQ', 'IVFPQ-RR',
                                   'IVFPQ-ONDISK', HNSW'}
    train_data : (float32)
        numpy.memmap or numpy.ndarray
    train_data_shape : list(int, int)
        Data shape (n, d). n is the number of items. d is dimension.
    use_gpu: (bool)
        If False, use CPU. Default is True.
    max_nitem_train : (int)
        Max number of items to be used for training index. Default is 1e7.

    Returns
    -------
    index : (faiss.swigfaiss_avx2.GpuIndex***)
        Trained FAISS index.

    References:

        https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

    """
    # GPU Setup
    if use_gpu:
        GPU_RESOURCES = faiss.StandardGpuResources()
        GPU_OPTIONS = faiss.GpuClonerOptions()
        GPU_OPTIONS.useFloat16 = True # use float16 table to avoid https://github.com/facebookresearch/faiss/issues/1178
        #GPU_OPTIONS.usePrecomputed = False
        #GPU_OPTIONS.indicesOptions = faiss.INDICES_CPU
    else:
        pass

    # Fingerprint dimension, d
    d = train_data_shape[1]

    # Build a flat (CPU) index
    index = faiss.IndexFlatL2(d) #

    mode = index_type.lower()
    print(f'Creating index: \033[93m{mode}\033[0m')
    if mode == 'l2':
        # Using L2 index
        pass
    elif mode == 'ivf':
        # Using IVF index
        nlist = 400
        index = faiss.IndexIVFFlat(index, d, nlist)
    elif mode == 'ivfpq':
        # Using IVF-PQ index
        code_sz = 64 # power of 2
        n_centroids = 256#
        nbits = 8  # nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
        index = faiss.IndexIVFPQ(index, d, n_centroids, code_sz, nbits)
    elif mode == 'ivfpq-rr':
        # Using IVF-PQ index + Re-rank
        code_sz = 64
        n_centroids = 256# 10:1.92ms, 30:1.29ms, 100: 0.625ms
        nbits = 8  # nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
        M_refine = 4
        nbits_refine = 4
        index = faiss.IndexIVFPQR(index, d, n_centroids, code_sz, nbits,
                                  M_refine, nbits_refine)
    elif mode == 'ivfpq-ondisk':
        if use_gpu:
            raise NotImplementedError(f'{mode} is only available in CPU.')
        raise NotImplementedError(mode)
    elif mode == 'hnsw':
        if use_gpu:
            raise NotImplementedError(f'{mode} is only available in CPU.')
        else:
            M = 16
            index = faiss.IndexHNSWFlat(d, M)
            index.hnsw.efConstruction = 80
            index.verbose = True
            index.hnsw.search_bounded_queue = True
    else:
        raise ValueError(mode.lower())

    # From CPU index to GPU index
    if use_gpu:
        print('Copy index to \033[93mGPU\033[0m.')
        index = faiss.index_cpu_to_gpu(GPU_RESOURCES, 0, index, GPU_OPTIONS)

    # Train index
    start_time = time.time()
    if len(train_data) > max_nitem_train:
        print('Training index using {:>3.2f} % of data...'.format(
            100. * max_nitem_train / len(train_data)))
        # shuffle and reduce training data
        sel_tr_idx = np.random.permutation(len(train_data))
        sel_tr_idx = sel_tr_idx[:max_nitem_train]
        index.train(train_data[sel_tr_idx,:])
    else:
        print('Training index...')
        index.train(train_data) # Actually do nothing for {'l2', 'hnsw'}
    print('Elapsed time: {:.2f} seconds.'.format(time.time() - start_time))

    # N probe
    index.nprobe = 40
    return index
