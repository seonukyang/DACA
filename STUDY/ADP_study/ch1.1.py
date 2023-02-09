import numpy as np
array1 = np.array([1,2,3])
print('array1 type:', type(array1))
print('array1[0] : ', array1[0], type(array1[0]))

array2 = np.array([[1,2,3],[4,5,6]])
print('array2 type : ', type(array2))
print('array2[0] : ', array2[0], type(array2[0]))
print('array2[0,0] : ', array2[0,0], type(array2[0,0]))
print('array2의 형태 :', array2.shape)

list1 = [1,2,3]
list1_1 = [1.1, 1.2, 1.3]

array1 = np.array(list1)
array1_1 = np.array(list1_1)
print('array1의 타입 : ', type(array1))
print('array1의 데이터 타입 : ', array1.dtype)
print('array1_1의 타입 : ', type(array1_1))
print('array1_1의 데이터 타입 : ', array1_1.dtype)
print(array1+array1_1)
array2 = np.array([list1,list1_1])
print(array2)
print(array2.dtype)

array_int = np.array([1,2,3])
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)

array_int1 = array_float.astype('int32')
print(array_int1, array_int1.dtype)

array_float1 = np.array([1.1,2.1,3.1])
array_int2 = array_float1.astype('int32')
print(array_int2, array_int2.dtype)