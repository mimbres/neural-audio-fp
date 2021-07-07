# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" mini_search_subroutines.py

A simple implementation of in-memory-search using NumPy and Tensorflow.

• Currently, this code is used only for mini search test in the training stage.  
• For large-scale experiments, it is strongly recommended to use the super fast
  FAISS-based implementation provided in this repo.
• In terms of speed and memory usage, [this code] < [FAISS-CPU] < [FAISS-GPU] 
  is the most efficient.

Functions:
    eval_mini_search(query, db,...):
        This function performs the mini-search test in the process of training.
    pairwise_distances_for_eval(emb_que, emb_db,...):
        Tensorflow implementation for calculating pairwise distance and its 
        substitutes. 
    conv_eye_func(x, s):  
        Used for calculating match score.

"""
import tensorflow as tf
import numpy as np


@tf.function
def pairwise_distances_for_eval(emb_que,
                                 emb_db,
                                 return_dotprod=False,
                                 squared=True):
    """
    Tensorflow implementation for calculating pairwise distance.
    Pairwise L2 squared distance matrix:        
        
      ||a - b||^2 = ||a||^2  + ||b||^2 - 2 dot_prod(a, b)
      
    Parameters
    ----------
    emb_que : (float) 
        Embeddings of queries with the shape (nQ, nAug, d).
        nQ is the number of queries, and nAug is the number of augmented (by 
        synthesis) versions in the queries. d is the embedding dimension.
        
    emb_db : (float) 
        Embeddings of DB with the shape (nD, d). nD is the number of items in
        DB, and d is the embedding dimension.
    
    return_dotprod : (bool), optional
        If True, output dot-product instead of euclidean distance. 
        The default is False.
    
    squared : (bool), optional
        If True, L2 squared distance. Else, euclidean distance.
        The default is True.

    Returns
    -------
    dists: (tf.Float32) 
        Pair-wise distance. Tensor of shape (n_augs, nQItem, nDItem, 1)
        
    """
    dot_product = tf.matmul(emb_que,
                            tf.transpose(emb_db))  # (nQItem,n_augs,nDItem)
    dot_product = tf.transpose(dot_product,
                               perm=[1, 0, 2])  # (n_augs, nQItem, nDItem)
    if return_dotprod:
        return tf.expand_dims(dot_product, 3)  #(n_augs, nQItem, nDItem, 1)
    else:
        pass

    # Get squared L2 norm for each embedding.
    que_sq = tf.reduce_sum(tf.square(emb_que), axis=2)  # (nItem, n_augs)
    que_sq = tf.transpose(que_sq, perm=[1, 0])  # (n_augs, nItem)
    db_sq = tf.reduce_sum(tf.square(emb_db), axis=1)  # (nItem,)
    db_sq = tf.reshape(db_sq, (1, -1))

    dists = tf.expand_dims(que_sq, 2) + tf.expand_dims(
        db_sq, 1) - 2.0 * dot_product
    dists = tf.maximum(dists, 0.0)  # Make sure every dist >= 0
    dists = tf.expand_dims(dists, 3)

    if not squared:
        mask = tf.cast(tf.equal(dists, 0.0), tf.float32)
        dists = dists + mask * 1e-16 # To prevent dividing by 0...
        dists = tf.sqrt(dists)
        dists = dists * (1.0 - mask)
    return dists


def conv_eye_func(x, s):
    """
    Convolution trick with tf.eye(.) filter for calculating sequence match 
    scores.

    Parameters
    ----------
    x : (tf.Float32)
        Pair-wise distance matrix.
        
    s : (int)
        Scope of input lengths to be tested. ex) [1, 3, 5, 9, 11, 19]

    Returns
    -------
    (tf.Float32)
        match score matrix.

    """
    conv_eye = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=[s, s],
        padding='valid',
        use_bias=False,
        kernel_initializer=tf.constant_initializer(
            np.eye(s).reshape((s, s, 1, 1))))
    conv_eye.trainable = False
    return conv_eye(x)


def mini_search_eval(query,
                     db,
                     scopes=[1, 3, 5, 9, 11, 19],
                     mode='argmin',
                     display=True,
                     gt_id_offset=0):
    """
    Tensorflow + NumPy implementation of mini-in-memory-search without data
    compression. Used for validation of fingerprinting model in training. 

    Parameters
    ----------
    query : (tf.float32)
        Embeddings of queries with the shape (nQ, nAug, d).
        nQ is the number of queries, and nAug is the number of augmented (by 
        synthesis) versions in the queries. d is the embedding dimension.
        
    db : (tf.float32)
        Embeddings of DB with the shape (nD, d). nD is the number of items in
        DB, and d is the embedding dimension.
        
    scopes : TYPE, optional
        DESCRIPTION. The default is [1, 3, 5, 9, 11, 19].
        
    mode : (str), optional
        'argmin': use for minimum distance search
        'argmax': use for maximum inner-product search    
        The default is 'argmin'.
        
    display : (bool), optional
        DESCRIPTION. The default is True.
        
    gt_id_offset : (int), optional
        experimental. The default is 0.

    Raises
    ------
    NotImplementedError
        raises when the value of 'mode' was neither 'argmin' nor 'argmax'.

    Returns
    -------
    (top1_acc, top3_acc, top10_acc) : (float, float, float)
        Top1, Top3, Top10 accuracies in percentage (%).
    mean_rank : (float)
        Mean rank.

    """
    n_augs = query.shape[1]
    n_scopes = len(scopes)
    
    # query = tf.constant(query.astype('float32'))
    # db = tf.constant(db.astype('float32'))
    
    # Compute pair-wise distance matrix, and convolve with tf.eye(scope)
    if mode == 'argmin':
        all_dists = pairwise_distances_for_eval(query, db,
                                                 squared=True).numpy()
    elif mode.lower() == 'argmax':
        all_dists = pairwise_distances_for_eval(query,
                                                 db,
                                                 return_dotprod=True).numpy()
    else:
        raise NotImplementedError(mode)

    mean_rank = np.zeros(n_scopes)
    top1_acc, top3_acc, top10_acc = np.zeros(n_scopes), np.zeros(
        n_scopes), np.zeros(n_scopes)
    for i, s in enumerate(scopes):
        conv_dists = conv_eye_func(all_dists, s).numpy()
        conv_dists = np.squeeze(conv_dists, 3)  # (n_augs, n_q, n_db)
        #print(i,s)
        
        # Mean-rank
        sorted = np.argsort(conv_dists, axis=2)
        if mode.lower() == 'argmax':
            sorted = sorted[:, :, ::-1]
        n_targets = conv_dists.shape[1]

        _sum_rank = 0
        for target_id in range(n_targets):
            gt_id = target_id + gt_id_offset  # this offset is required for large-scale search only
            _, _rank = np.where(sorted[:, target_id, :] == gt_id)
            _sum_rank += np.sum(_rank) / n_augs
        mean_rank[i] = _sum_rank / n_targets

        # Top1,Top3,Top10 Acc
        _n_corrects_top1, _n_corrects_top3, _n_corrects_top10 = 0, 0, 0
        for target_id in range(n_targets):
            gt_id = target_id + gt_id_offset  # this offset is required for large-scale search only
            _n_corrects_top1 += np.sum(sorted[:, target_id,
                                              0] == gt_id) / n_augs  #4
            _n_corrects_top3 += np.sum(
                sorted[:, target_id, :3] == gt_id) / n_augs  #4
            _n_corrects_top10 += np.sum(
                sorted[:, target_id, :10] == gt_id) / n_augs  #4
        top1_acc[i] = _n_corrects_top1 / n_targets 
        top3_acc[i] = _n_corrects_top3 / n_targets
        top10_acc[i] = _n_corrects_top10 / n_targets
    top1_acc *= 100.
    top3_acc *= 100.
    top10_acc *= 100.
    
    if display:
        color_cyan = '\033[36m'
        color_def = '\033[0m'
        line_int = '{:^6}\t' * len(scopes) 
        line_float = '{:>4.2f}\t' * len(scopes) 
        print(color_cyan + 'Scope:\t', line_int.format(*scopes), color_def)
        print(color_cyan +'T1acc:\t' + color_def, line_float.format(*top1_acc))
        print(color_cyan +'mRank:\t' + color_def, line_float.format(*mean_rank))


    return (top1_acc, top3_acc, top10_acc), mean_rank
