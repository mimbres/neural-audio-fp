# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" online_triplet_loss.py
 
Triplet margin loss with online triplet mining, adopted in "FaceNet" and 
"Now-playing".

USAGE:
    
    loss_obj = OnlineTripletLoss(bsz=100, n_anchor=20, n_pos_per_anchor=4,
                                 use_anc_as_pos=True)
    loss = loss_obj.compute_triplet_loss_v2(...)

References:
    • F Schroff et al. "FaceNet: A Unified Embedding for Face Recognition and
      Clustering" CVPR 2015 https://arxiv.org/abs/1503.03832
    • G Feller et al. "Now Playing: Continuous low-power music recognition"
      NeurIPS 2017 Workshop: Machine Learning on the Phone 
      https://arxiv.org/abs/1711.10958
    • Implemented based on https://github.com/omoindrot/tensorflow-triplet-loss


NOTE: This code is taken from my old experiment to replicate "Now-playing" in 
      winter 2019. This part will be combined in the next update.

"""
import tensorflow as tf
import numpy as np


class OnlineTripletLoss():
    def __init__(self,
                 bsz=int(),
                 n_anchor=int(),
                 n_pos_per_anchor=int(),
                 use_anc_as_pos=True):
        # Variables
        self.bsz = bsz # Batch size
        self.n_anchor = n_anchor # Total number of anchor samples in batch
        self.n_pos_per_anchor = n_pos_per_anchor # Number of positive samples per each anchor
        self.use_anc_as_pos = use_anc_as_pos # Using each original anchor sample as a member of positive samples
        
        # Anchor-positive & Anchor-engative masks
        self.ap_mask = self._get_anchor_positive_mask_v2()
        self.an_mask = self._get_anchor_negative_mask_v2()
        #self.ap_mask_bin = tf.cast(self.ap_mask, tf.bool)
        self.an_mask_bin = tf.cast(self.an_mask, tf.bool)
        self.gt_mask = tf.abs(1 - self.an_mask) 
        self.mask_shape = self.ap_mask.shape
        
        # Number of positive and negative elements per each anchor (to be used as a normalization factor)
        self.num_ap_elem_per_anc = tf.constant(np.sum(self.ap_mask, axis=1).astype(np.float32)) # (A,)
        self.num_an_elem_per_anc = tf.constant(np.sum(self.an_mask, axis=1).astype(np.float32)) # (A,)


    #---- Numpy functions for generating masks --------------------------------
    def _get_anchor_positive_mask_v2(self):
        n_pos = self.n_anchor * self.n_pos_per_anchor # Here, n_pos does not count anchor-anchor datapoints.
        
        if self.use_anc_as_pos:
            mask = np.zeros((self.n_anchor, n_pos + self.n_anchor)) # (nA, bsz)
        else:
            mask = np.zeros((self.n_anchor, n_pos))
        
        for anchor in range(self.n_anchor):
            mask[anchor, anchor * self.n_pos_per_anchor: (anchor + 1) * self.n_pos_per_anchor] = 1    
        return tf.constant(mask.astype(np.float32))
    
    
    def _get_anchor_negative_mask_v2(self):
        mask = self._get_anchor_positive_mask_v2()
        
        if self.use_anc_as_pos:
            mask = tf.concat((mask[:, :self.n_anchor * self.n_pos_per_anchor], tf.eye(self.n_anchor)), axis=1)
        return (1 - mask)
        

    # ---- Tensorflow functions for calculation of pariwise distances ---------
    # @tf.function
    # def _pairwise_distances_v2(self, emb_anc, emb_pos, use_anc_as_pos=True, squared=False):
    #     """Compute the 2D matrix of distances between anchors and positive embeddings.
        
    #     Args:
    #         emb_anc: tensor of shape (nA, Q)
    #         emb_pos: tensor of shape (nP, Q)
    #         NOTE: {emb_anc, emb_pos} must be L2-normalized vectors with axis=1. 
            
    #     Returns:
    #         if use_anc_as_pos:
    #             pairwise_distances: tensor of shape (nA, nP + nA)
    #         else:
    #             pairwise_distances: tensor of shape (nA, nP)
        
    #     """
    #     if use_anc_as_pos:
    #         emb_pos = tf.concat((emb_pos, emb_anc), axis=0) # (A+P, Q)
    #     else:
    #         pass;
    #     dot_product = tf.matmul(emb_anc, tf.transpose(emb_pos)) # (A, A+P)
        
    #     # Get squared L2 norm for each embedding.
    #     a_sq = tf.reduce_sum(tf.square(emb_anc), axis=1)# (A, 1) 
    #     p_sq = tf.reduce_sum(tf.square(emb_pos), axis=1)# (P, 1) or (A+P, 1)
        
    #     """ Pairwise squared distance matrix:        
        
    #         ||a - b||^2 = ||a||^2  + ||b||^2 - - 2 <a, b>
        
    #     """
        
    #     dists = (tf.expand_dims(a_sq, 1) + tf.expand_dims(p_sq, 0)) - 2.0 * dot_product
    #     dists = tf.maximum(dists, 0.0) # Make sure every dist >= 0
    
    #     if not squared:
    #         # To prevent -inf gradients where dist=0.0, we add small epsilons
    #         mask = tf.cast(tf.equal(dists, 0.0), tf.float32)
    #         dists = dists + mask * 1e-16
    #         dists = tf.sqrt(dists)
    #         dists = dists * (1.0 - mask)
    #     return dists


    @tf.function
    def _pairwise_dotprod(self, emb_anc, emb_pos, use_anc_as_pos=True):
        """ Compute 2D cosine-similarity matrix  
        
        Args:
            emb_anc: tensor of shape (nA, d), d is dimension of embeddings
            emb_pos: tensor of shape (nP, d)
            NOTE: {emb_anc, emb_pos} must be L2-normalized vectors with axis=1.     
        
        Returns:
            if use_anc_as_pos:
                pairwise_distances: tensor of shape (nA, nP + nA)
            else:
                pairwise_distances: tensor of shape (nA, nP)
        
        """
        if use_anc_as_pos:
            emb_pos = tf.concat((emb_pos, emb_anc), axis=0) # (A+P, Q)
        else:
            pass;
        
        return tf.matmul(emb_anc, tf.transpose(emb_pos))


    @tf.function
    def _pairwise_distances_v2_fast(self, emb_anc, emb_pos, use_anc_as_pos=True, squared=False):
        dists = self._pairwise_dotprod(emb_anc=emb_anc,
                                     emb_pos=emb_pos,
                                     use_anc_as_pos=use_anc_as_pos)
        dists = 2. * (1 - dists)
        if not squared:
            # To prevent -inf gradients where dist=0.0, we add small epsilons
            mask = tf.cast(tf.equal(dists, 0.0), tf.float32)
            dists = dists + mask * 1e-16
            dists = tf.sqrt(dists)
            dists = dists * (1.0 - mask)
        
        return dists
        
        
    # ---- Loss Functions -----------------------------------------------------
    @tf.function
    def compute_triplet_loss_v2(self, emb_anchor, emb_pos, mode, margin=1.0,
                              use_anc_as_pos=True, squared=False):
        """
        mode:
            "all"
            "all-balanced"
            "hardest"
            "semi-hard"
            
        """
        # Get a pairwise distance matrix
        #pairwise_dist = self._pairwise_distances_v2(emb_anchor, emb_pos, use_anc_as_pos, squared) # (A, P)
        pairwise_dist = self._pairwise_distances_v2_fast(emb_anchor, emb_pos, use_anc_as_pos, squared)
        # Calculate Pos/Neg distances
        ap_dists = pairwise_dist * self.ap_mask
        
        if mode == "all":
            an_dists = pairwise_dist * self.an_mask
            loss = tf.maximum(ap_dists - an_dists + margin, 0.)
            loss = tf.reduce_mean(loss)
        elif mode == "all-balanced":
            ap_dists = tf.divide(tf.reduce_sum(ap_dists, axis=1), self.num_ap_elem_per_anc) 
            an_dists = pairwise_dist * self.an_mask
            an_dists = tf.divide(tf.reduce_sum(an_dists, axis=1), self.num_an_elem_per_anc)
            loss = tf.maximum(ap_dists - an_dists + margin, 0.)
            loss = tf.reduce_mean(loss)
        elif mode == "hardest":
            ap_dists = tf.reduce_max(ap_dists, axis=1) 
            an_dists = tf.reduce_min(pairwise_dist * self.an_mask, axis=1)
            loss = tf.maximum(ap_dists - an_dists + margin, 0.)
            loss = tf.reduce_mean(loss)
        elif mode == "semi-hard":
            # ap_dists: Tiled hardest anchor-positive distances.
            ap_dists = tf.reduce_max(ap_dists, axis=1, keepdims=True) * tf.ones([1, self.mask_shape[1]])
            loss = (ap_dists - pairwise_dist + margin) * self.an_mask
            loss = tf.maximum(loss, 0.) # Neglect easy triplets
            #num_active_triplets = tf.reduce_sum(tf.cast(tf.greater(loss, 0.), tf.float32))
            loss = tf.reduce_mean(loss) #/ num_active_triplets # <-- fixed 09.27
        else:
            raise NotImplementedError(mode)

        return loss 
    

def test_mask():
    # Display generated masks
    import matplotlib.pyplot as plt
    loss_obj = OnlineTripletLoss(bsz=40, n_anchor=8, n_pos_per_anchor=4,
                                     use_anc_as_pos=True)
    ap_mask = loss_obj._get_anchor_positive_mask_v2()
    plt.figure()
    plt.imshow(ap_mask); plt.title('Anchor-positive-mask: bsz=40, nAnchor=8, nPosPerAnchor=4')

    an_mask = loss_obj._get_anchor_negative_mask_v2()
    plt.figure()
    plt.imshow(an_mask); plt.title('Anchor-negative-mask: bsz=40, nAnchor=8, nPosPerAnchor=4')
    

def test_pairwise_dist():
    emb_anc = tf.random.uniform((8,64))
    emb_pos = tf.random.uniform((32,64))
    emb_anc = tf.math.l2_normalize(emb_anc, axis=1)
    emb_pos = tf.math.l2_normalize(emb_pos, axis=1)
    
    loss_obj = OnlineTripletLoss(bsz=40, n_anchor=8, n_pos_per_anchor=4, use_anc_as_pos=True)
    dist1 = loss_obj._pairwise_distances_v2(emb_anc, emb_pos, use_anc_as_pos=True, squared=True) 
    dist2 = 2 * (1 - loss_obj._pairwise_dotprod(emb_anc, emb_pos, use_anc_as_pos=True))
    dist3 = loss_obj._pairwise_distances_v2_fast(emb_anc, emb_pos, use_anc_as_pos=True, squared=True)  
    
    assert(tf.reduce_sum(dist1-dist2) < 0.0000001)
    assert(tf.reduce_sum(dist1-dist3) < 0.0000001)
    return       
    # dist1: L2 distance 19.2ms
    # dist2: fast L2_v2 with dot-product 9.26ms
    