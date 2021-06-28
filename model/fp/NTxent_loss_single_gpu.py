# -*- coding: utf-8 -*-
"""NTxent_loss_single_gpu.py

Simple implementation of Normalized Temperature Crossentropy loss for 
single GPU.


    Input batch for FP training:
    - We assume a batch of ordered embeddings as {a0, a1,...b0, b1,...}.
    - In SimCLR paper, a(i) and b(i) are augmented samples from the ith
      original sample.
    - In our Fingerprinter, we assume a(i) is ith original sample, while b(i) 
      is augmented samples from a(i).
    - In any case, input embeddings should be split by part a and b.
    
    How is it different from SimCLR author's code?
    - The drop_diag() part gives the better readability, and is coceptually
      more making sense (in my opinion).
    - Other than that, it is basically equivalent.
    
    Why can't we use this code for multi-GPU or TPUs?
    - drop_diag() part will not work properly there.
    - So I provide NTxent_fp_loss_tpu.py for multi-GPU and TPUs.
        
"""
import tensorflow as tf


class NTxentLoss():

    def __init__(self,
                 n_org=int(),
                 n_rep=int(), 
                 tau=0.05,
                 **kwargs
                 ):
        """Init."""
        self.n_org = n_org
        self.n_rep = n_rep
        self.tau = tau
        
        """Generate temporal labels and diag masks."""
        self.labels = tf.one_hot(tf.range(n_org), n_org * 2 - 1)
        self.mask_not_diag = tf.constant(tf.cast(1 - tf.eye(n_org), tf.bool))
        
    
    @tf.function 
    def drop_diag(self, x):
        x = tf.boolean_mask(x, self.mask_not_diag)
        return tf.reshape(x, (self.n_org, self.n_org-1))
    
    
    @tf.function 
    def compute_loss(self, emb_org, emb_rep):
        """NTxent Loss function for neural audio fingerprint.
        
        NOTE1: all input embeddings must be L2-normalized... 
        NOTE2: emb_org and emb_rep must be even number.
        
        Args:
            emb_org: tensor of shape (nO, d), nO is the number of original samples, and d is dimension of embeddings. 
            emb_rep: tensor of shape (nR, d)        
                    
        Returns:
	    (loss, sim_mtx, labels)
        
        """
        ha, hb = emb_org, emb_rep # assert(len(emb_org)==len(emb_rep))
        logits_aa = tf.matmul(ha, ha, transpose_b=True) / self.tau
        logits_aa = self.drop_diag(logits_aa) # modified
        logits_bb = tf.matmul(hb, hb, transpose_b=True) / self.tau
        logits_bb = self.drop_diag(logits_bb) # modified
        logits_ab = tf.matmul(ha, hb, transpose_b=True) / self.tau
        logits_ba = tf.matmul(hb, ha, transpose_b=True) / self.tau
        loss_a = tf.compat.v1.losses.softmax_cross_entropy(
            self.labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.compat.v1.losses.softmax_cross_entropy(
            self.labels, tf.concat([logits_ba, logits_bb], 1))
        return loss_a + loss_b, tf.concat([logits_ab, logits_aa], 1), self.labels


# Unit-test
def test_loss():
    feat_dim = 5
    n_org, n_rep = 3, 3   # NOTE: usually n_org = n_rep. because we always prepare pairwise. Batchsize is 6
    tau = 0.05 # temperature
    emb_org = tf.random.uniform((n_org, feat_dim)) # this should be [org1, org2, org3]
    emb_rep = tf.random.uniform((n_rep, feat_dim)) # this should [rep1, rep2, rep3] 
    
    loss_obj = NTxentLoss(n_org=n_org, n_rep=n_rep, tau=tau)
    loss, simmtx_upper_half, _ = loss_obj.compute_loss(emb_org, emb_rep)
    