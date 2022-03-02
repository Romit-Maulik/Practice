import numpy as np
import pickle
import matplotlib.pyplot as plt

file = open("Train_Predictions.pkl",'rb')
object_file = pickle.load(file)
file.close()


for i in range(4):

    true = object_file[0][i]
    predicted = object_file[1][i]

    print(true)

    # print(true.shape)
    # print(predicted.shape)

    # exit()

    plt.figure()
    plt.plot(true[:,-1,0],label='True')
    plt.plot(predicted[:,-1,0],label='Predicted')
    plt.legend()

    plt.figure()
    plt.plot(true[:,-1,1],label='True')
    plt.plot(predicted[:,-1,1],label='Predicted')
    plt.legend()


    plt.figure()
    plt.plot(true[:,-1,2],label='True')
    plt.plot(predicted[:,-1,2],label='Predicted')
    plt.legend()
plt.show()
