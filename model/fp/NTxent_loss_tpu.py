# -*- coding: utf-8 -*-
""" NTxent_loss_tpu.py """
import tensorflow as tf


class NTxentLoss():
    """
    Simple implementation of Normalized Temperature Crossentropy loss for 
    multiple GPUs and TPUs.
    
    How should the input batch be prepared for FP training?
    • We assume a batch of ordered embeddings as {a0, a1,...b0, b1,...}.
    • In SimCLR paper, a(i) and b(i) are augmented samples from the ith
      original sample.
    • In this work, we assume a(i) to be ith original sample, while b(i) 
      to be augmented samples from a(i).
    • In any case, input embeddings should be split by part a and b.
    
    Why not provide TPUs code in this release?
    • Training with TPUs requires a new data pipeline.
    • In particular, the augmentation pipeline must be implemented based on
      fully GPU/TPUs tensors. I am working on new code for that. Since I
      renewed augmentation methods there, it will be unsuitable for the purpose
      of repoducing the ICASSP paper's result.
    
    Why are multiple GPUs or TPUs important? What are the benefits?
    • The larger the batch size, the better the performance in contrastive 
      learning.
    
    References:
        https://www.tensorflow.org/api_docs/python/tf/distribute
    
    """
    def __init__(self, local_bsz=5, tau=0.1, LARGE_NUM=1e9):
        self.local_bsz = local_bsz
        self.n_a = local_bsz // 2
        self.n_b = local_bsz // 2
        self.tau = tau
        self.LARGE_NUM = LARGE_NUM

        
    @tf.function    
    def get_labels_and_masks(self, ctx=None):
        if ctx is None: # Single GPU
            labels = tf.one_hot(tf.range(self.n_a), self.local_bsz) 
            diag_masks = tf.eye(self.n_a)
        else: # TPU
            n_replicas = ctx.num_replicas_in_sync
            rep_id = ctx.replica_id_in_sync_group
            labels_idx = tf.range(self.n_a) + rep_id * self.n_a
            # labels_idx = tf.range(self.n_a) + rep_id * self.local_bsz # FIX 0714
            labels = tf.one_hot(labels_idx, self.local_bsz * n_replicas)
            diag_masks = tf.one_hot(labels_idx, self.n_a * n_replicas)
        return labels, diag_masks
    
    
    @tf.function(experimental_relax_shapes=True)
    def tpu_cross_replica_concat(self, tensor, ctx=None):
        """ input 'tensor' is within per replica context """
        
        if ctx is None or ctx.num_replicas_in_sync <=1:
            # bypass
            return tensor
        else:
            n_replicas = ctx.num_replicas_in_sync
            rep_id = ctx.replica_id_in_sync_group

            """
            Fill per-replica tensor in place:
                https://www.tensorflow.org/api_docs/python/tf/scatter_nd
            """
            ext_tensor = tf.scatter_nd(indices=[[rep_id]],
                                       updates=[tensor],
                                       shape=[n_replicas] + tensor.shape.as_list()
                                      )

            """
            Cross-replica sum:
                https://www.tensorflow.org/api_docs/python/tf/distribute/ReplicaContext#all_reduce
            """
            ext_tensor = ctx.all_reduce(reduce_op=tf.distribute.ReduceOp.SUM, value=ext_tensor)
            
            """
            Flatten the replica dimension:
                - the first dimension size will be: tensor.shape[0] * num_replicas
            """
        return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:]) # (N_REP*BSZ, D)
    
    
    @tf.function(experimental_relax_shapes=True) 
    def loss_fn(self, embeddings, use_tpu):
        # Split a, b (in FP: a=org, b=aug(org))
        ha, hb = tf.split(embeddings, 2, axis=0)
        
        if use_tpu==True:
            # Get {ha_large, hb_large} with cross-replica context
            ctx = tf.distribute.get_replica_context()
        else:
            ctx = None
            
        if ctx is None or ctx.num_replicas_in_sync <=1:
            ha_large = ha
            hb_large = hb
        else:
            # {ha_large, hb_large}: Cross-replica concat of {ha, hb}
            # n_replicas = ctx.num_replicas_in_sync
            ha_large = self.tpu_cross_replica_concat(ha, ctx)
            hb_large = self.tpu_cross_replica_concat(hb, ctx)
        
        
        # Get pseudo labels and diagonoal masks
        labels, diag_masks = self.get_labels_and_masks(ctx)
        
        
        # Separate Dot-product by pos/neg cases        
        logits_aa = tf.matmul(ha, ha_large, transpose_b=True) / self.tau
        logits_aa = logits_aa - diag_masks * self.LARGE_NUM
        logits_bb = tf.matmul(hb, hb_large, transpose_b=True) / self.tau
        logits_bb = logits_bb - diag_masks * self.LARGE_NUM
        logits_ab = tf.matmul(ha, hb_large, transpose_b=True) / self.tau
        logits_ba = tf.matmul(hb, ha_large, transpose_b=True) / self.tau
#         print("ha:", ha)
#         print("ha_large:", ha_large)
#         print("logits_aa :", logits_aa)
#         print("labels: ", labels)
#         print("diag_masks :", diag_masks)
        
        loss_a = tf.nn.softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.nn.softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_ba, logits_bb], 1))

        # loss_a = tf.compat.v1.losses.softmax_cross_entropy(
        #         labels, tf.concat([logits_ab, logits_aa], 1))
        # loss_b = tf.compat.v1.losses.softmax_cross_entropy(
        #         labels, tf.concat([logits_ba, logits_bb], 1))    
        return loss_a + loss_b, tf.concat([logits_ab, logits_aa], 1), labels
        
        
        
# Unit-test
def test_loss():
    feat_dim = 5
    n_org, n_rep = 3, 6
    emb_org = tf.random.uniform((n_org, feat_dim))
    emb_rep = tf.random.uniform((n_rep, feat_dim))
    
    loss_obj = NTxentLoss(local_bsz=1000)
    loss, simmtx_upper_half, _ = loss_obj.loss_fn(emb_org, emb_rep)
