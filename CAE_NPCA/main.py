import numpy as np
np.random.seed(10)
import tensorflow as tf
tf.random.set_seed(10)
import matplotlib.pyplot as plt

from CAE_Model import CAE_Model


if __name__ == '__main__':
    dataset = np.load('All_Snapshot_Data.npy')
    num_latent = 8
    npca = True
    model = CAE_Model(dataset,num_latent,npca)

    model.train_model()
    model.restore_model()
    model.model_inference()


