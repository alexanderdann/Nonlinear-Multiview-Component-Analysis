import tensorflow as tf
import numpy as np
from models import MiniMaxCCA, CNNdecoder, CNNencoder, CNNDAE




if __name__ == '__main__':
    # Input data is of dimension M_q * N
    print(tf.version)