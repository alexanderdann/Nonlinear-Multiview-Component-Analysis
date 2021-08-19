import tensorflow as tf
import numpy as np


def update_U(model, eval_helper, z_dim):
    ZW_tmp = []

    for batch_idx, view1, view2, _ in enumerate(eval_helper):
        shared, _ = model.encode(tf.constant(view1), tf.constant(view2))
        ZW_tmp.append(tf.concat(shared, 1))

    ZW_tmp = tf.concat(tf.constant(ZW_tmp), 0)
    ZW_tmp = ZW_tmp - tf.math.reduce_mean(ZW_tmp, 0, keepdims=True)

    tmp = []
    for ind in range(2):
        tmp.append(ZW_tmp[:, ind * z_dim:(ind + 1) * z_dim])

    ZW = tf.stack(tmp, axis=2)

    P, D, Q = tf.linalg.svd(ZW)
    Q_T = tf.transpose(Q)
    U_nofac = tf.matmul(P, Q_T)
    U = U_nofac * tf.sqrt(U_nofac.shape[0])
    return U



