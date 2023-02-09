import numpy as np
#3) 배열의 인덱싱과 슬라이싱
#인덱싱
a1 = np.array([0,10,20,30,40,50])
print(type(a1))
print(a1[4])
a1[5] = 70
print(a1)
print(a1[[1,3,4]])

#슬라이싱
b1 = np.arange(6)
print(b1)
print(b1[1:4])
b1[2:5] = np.array([10,20,30])
print(b1)

#4) 배열의 차원 변경하기
Vector = np.arange(10)
Matrix = Vector.reshape(2,5)
print(Matrix)