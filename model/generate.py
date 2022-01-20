# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" generate.py """
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from model.dataset import Dataset
from model.fp.melspec.melspectrogram import get_melspec_layer
from model.fp.nnfp import get_fingerprinter


def build_fp(cfg):
    """ Build fingerprinter """
    # m_pre: log-power-Mel-spectrogram layer, S.
    m_pre = get_melspec_layer(cfg, trainable=False)

    # m_fp: fingerprinter g(f(.)).
    m_fp = get_fingerprinter(cfg, trainable=False)
    return m_pre, m_fp


def load_checkpoint(checkpoint_root_dir, checkpoint_name, checkpoint_index,
                    m_fp):
    """ Load a trained fingerprinter """
    # Create checkpoint
    checkpoint = tf.train.Checkpoint(model=m_fp)
    checkpoint_dir = checkpoint_root_dir + f'/{checkpoint_name}/'
    c_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir,
                                           max_to_keep=None)

    # Load
    if checkpoint_index == None:
        tf.print("\x1b[1;32mArgument 'checkpoint_index' was not specified.\x1b[0m")
        tf.print('\x1b[1;32mSearching for the latest checkpoint...\x1b[0m')
        latest_checkpoint = c_manager.latest_checkpoint
        if latest_checkpoint:
            checkpoint_index = int(latest_checkpoint.split(sep='ckpt-')[-1])
            status = checkpoint.restore(latest_checkpoint)
            status.expect_partial()
            tf.print(f'---Restored from {c_manager.latest_checkpoint}---')
        else:
            raise FileNotFoundError(f'Cannot find checkpoint in {checkpoint_dir}')
    else:
        checkpoint_fpath = checkpoint_dir + 'ckpt-' + str(checkpoint_index)
        status = checkpoint.restore(checkpoint_fpath) # Let TF to handle error cases.
        status.expect_partial()
        tf.print(f'---Restored from {checkpoint_fpath}---')
    return checkpoint_index


def prevent_overwrite(key, target_path):
    if (key == 'dummy_db') & os.path.exists(target_path):
        answer = input(f'{target_path} exists. Will you overwrite (y/N)?')
        if answer.lower() not in ['y', 'yes']: sys.exit()


def get_data_source(cfg, source_root_dir, skip_dummy):
    dataset = Dataset(cfg)
    ds = dict()
    if source_root_dir:
        ds['custom_source'] = dataset.get_custom_db_ds(source_root_dir)
    else:
        if skip_dummy:
            tf.print("Excluding \033[33m'dummy_db'\033[0m from source.")
            pass
        else:
            ds['dummy_db'] = dataset.get_test_dummy_db_ds()

        if dataset.datasel_test_query_db in ['unseen_icassp', 'unseen_syn']:
            ds['query'], ds['db'] = dataset.get_test_query_db_ds()
        else:
            raise ValueError(dataset.datasel_test_query_db)

    tf.print(f'\x1b[1;32mData source: {ds.keys()}\x1b[0m',
             f'{dataset.datasel_test_query_db}')
    return ds


@tf.function
def test_step(X, m_pre, m_fp):
    """ Test step used for generating fingerprint """
    # X is not (Xa, Xp) here. The second element is reduced now.
    m_fp.trainable = False
    return m_fp(m_pre(X))  # (BSZ, Dim)


def generate_fingerprint(cfg,
                         checkpoint_name,
                         checkpoint_index,
                         source_root_dir,
                         output_root_dir,
                         skip_dummy):
    """
    After run, the output (generated fingerprints) directory will be:
      .
      └──logs
         └── emb
             └── CHECKPOINT_NAME
                 └── CHECKPOINT_INDEX
                     ├── db.mm
                     ├── db_shape.npy
                     ├── dummy_db.mm
                     ├── dummy_db_shape.npy
                     ├── query.mm
                     └── query_shape.npy
    """
    # Build and load checkpoint
    m_pre, m_fp = build_fp(cfg)
    checkpoint_root_dir = cfg['DIR']['LOG_ROOT_DIR'] + 'checkpoint/'
    checkpoint_index = load_checkpoint(checkpoint_root_dir, checkpoint_name,
                                       checkpoint_index, m_fp)

    # Get data source
    """ ds = {'key1': <Dataset>, 'key2': <Dataset>, ...} """
    ds = get_data_source(cfg, source_root_dir, skip_dummy)

    # Make output directory
    if output_root_dir:
        output_root_dir = output_root_dir + f'/{checkpoint_name}/{checkpoint_index}/'
    else:
        output_root_dir = cfg['DIR']['OUTPUT_ROOT_DIR'] + \
            f'/{checkpoint_name}/{checkpoint_index}/'
    os.makedirs(output_root_dir, exist_ok=True)
    if not skip_dummy:
        prevent_overwrite('dummy_db', f'{output_root_dir}/dummy_db.mm')

    # Generate
    sz_check = dict() # for warning message
    for key in ds.keys():
        bsz = int(cfg['BSZ']['TS_BATCH_SZ'])  # Do not use ds.bsz here.
        # n_items = len(ds[key]) * bsz
        n_items = ds[key].n_samples
        dim = cfg['MODEL']['EMB_SZ']
        """
        Why use "memmap"?

        • First, we need to store a huge uncompressed embedding vectors until
          constructing a compressed DB with IVF-PQ (using FAISS). Handling a
          huge ndarray is not a memory-safe way: "memmap" consume 0 memory.

        • Second, Faiss-GPU does not support reconstruction of DB from
          compressed DB (index). In eval/eval_faiss.py, we need uncompressed
          vectors to calaulate sequence-level matching score. The created
          "memmap" will be reused at that point.

        Reference:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

        """
        # Create memmap, and save shapes
        assert n_items > 0
        arr_shape = (n_items, dim)
        arr = np.memmap(f'{output_root_dir}/{key}.mm',
                        dtype='float32',
                        mode='w+',
                        shape=arr_shape)
        np.save(f'{output_root_dir}/{key}_shape.npy', arr_shape)

        # Fingerprinting loop
        tf.print(
            f"=== Generating fingerprint from \x1b[1;32m'{key}'\x1b[0m " +
            f"bsz={bsz}, {n_items} items, d={dim}"+ " ===")
        progbar = Progbar(len(ds[key]))

        """ Parallelism to speed up preprocessing------------------------- """
        enq = tf.keras.utils.OrderedEnqueuer(ds[key],
                                              use_multiprocessing=True,
                                              shuffle=False)
        enq.start(workers=cfg['DEVICE']['CPU_N_WORKERS'],
                  max_queue_size=cfg['DEVICE']['CPU_MAX_QUEUE'])
        i = 0
        while i < len(enq.sequence):
            progbar.update(i)
            X, _ = next(enq.get())
            emb = test_step(X, m_pre, m_fp)
            arr[i * bsz:(i + 1) * bsz, :] = emb.numpy() # Writing on disk.
            i += 1
        progbar.update(i, finalize=True)
        enq.stop()
        """ End of Parallelism-------------------------------------------- """

        tf.print(f'=== Succesfully stored {arr_shape[0]} fingerprint to {output_root_dir} ===')
        sz_check[key] = len(arr)
        arr.flush(); del(arr) # Close memmap

    if 'custom_source' in ds.keys():
        pass;
    elif sz_check['db'] != sz_check['query']:
        print("\033[93mWarning: 'db' and 'qeury' size does not match. This can cause a problem in evaluataion stage.\033[0m")
    return
