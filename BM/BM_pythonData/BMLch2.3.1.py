import numpy as np
data1 = [0,1,2,3,4,5]
a1 = np.array(data1)
print(a1, type(a1))
print(a1.dtype)

#1. 1부터 10까지 범위 안에서 간격 2를 갖는 배열 생성
a1 = np.arange(start = 1, stop = 10, step = 2)
print(a1)

#2. 0부터 10까지 범위 안에서 간격 1을 갖는 배열 생성
a2 = np.arange(10)
print(a2)

#2) 배열 연산하기
arr1 = np.array([10,15,20,30])
arr2 = np.array([1,1,2,7])
print(arr1+arr2)
print(arr1*arr2)
print(arr1/arr2)


print(type(a1))