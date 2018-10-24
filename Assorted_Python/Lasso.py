import numpy as np
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
inputs = np.zeros((100,3),dtype='double')
outputs = np.zeros((100,),dtype='double')

for i in range(np.shape(inputs)[0]):
    inputs[i, 0] = np.random.uniform(0.0, 1.0)
    inputs[i, 1] = np.random.uniform(0.0, 1.0)
    inputs[i, 2] = np.random.uniform(0.0, 1.0)

    outputs[i] = 3.0*inputs[i,0]+200.0*inputs[i,1]+inputs[i,2]

clf.fit(inputs,outputs)

print(clf.coef_)