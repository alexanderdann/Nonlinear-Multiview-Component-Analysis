import tensorflow as tf
import numpy as np
import scipy

def dist(S, U, N):
    assert tf.shape(S)[1] == N
    assert tf.shape(U)[1] == N

    tmp = tf.matmul(tf.matmul(tf.transpose(S), tf.linalg.inv(tf.matmul(S, tf.transpose(S)))), S)
    P_S = tf.cast(tf.subtract(tf.eye(N, dtype=tf.float64), tmp), dtype=tf.float32)
    Q_S = tf.convert_to_tensor(scipy.linalg.orth(U.numpy().T), dtype=tf.float32)
    PQ = tf.matmul(P_S, Q_S)

    return np.linalg.norm(PQ.numpy(), ord=2)

def PCC_Matrix(view1, view2, observations):
    assert tf.shape(view1)[1] == observations
    assert tf.shape(view1)[1] == tf.shape(view2)[1]

    calc_cov = np.zeros([tf.shape(view1)[0], tf.shape(view2)[0]])

    for dim1 in range(tf.shape(view1)[0]):
        for dim2 in range(tf.shape(view2)[0]):
            mu_1 = tf.reduce_mean(view1[dim1])
            mu_2 = tf.reduce_mean(view2[dim2])
            sigma_1 = tf.math.sqrt(tf.reduce_sum([(x - mu_1) ** 2 for x in view1[dim1]]))
            sigma_2 = tf.math.sqrt(tf.reduce_sum([(x - mu_2) ** 2 for x in view2[dim2]]))
            tmp = tf.reduce_sum([(view1[dim1][i]-mu_1)*(view2[dim2][i]-mu_2) for i in range(observations)])/(sigma_1*sigma_2)
            calc_cov[dim1, dim2] = np.abs(tmp.numpy())

    return calc_cov, tf.shape(view1)[0], tf.shape(view2)[0]


