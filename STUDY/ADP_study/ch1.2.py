import pandas as pd
import numpy as np
sequence_array = np.arange(10)
print(sequence_array)

zero_array = np.zeros((3,2), dtype='int32')
print(zero_array)
print(zero_array.dtype, zero_array.shape)

one_array = np.ones((3,2))
print(one_array)
print(one_array.dtype, one_array.shape)

array1 = np.arange(10)
print('array1:\n', array1)

array2 = array1.reshape(2,5)
print('array2:\n', array2)

array3 = array1.reshape(5,2)
print('array3:\n',array3)

array2 = array1.reshape(-1,5)
print('array2:\n', array2)

array3 = array1.reshape(5,-1)
print('array3:\n',array3)

array1 = np.arange(8)
array3d = array1.reshape((2,2,2))
print('array3d:\n', array3d)
print('array3d list:\n', array3d.tolist())

array5 = array3d.reshape(-1,1)
print('array5:\n', array5.tolist())
print('array5 shape:', array5.shape)

array6 = array1.reshape(-1,1)
print('array6:\n', array6.tolist())
print('array6 shape : ', array6.shape)
