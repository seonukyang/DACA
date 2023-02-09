import numpy as np

Vector = np.arange(10)
Matrix = Vector.reshape(2,5) #(2행 5열)
print(Matrix)

Vector1 = np.arange(18)
Matrix1 = Vector1.reshape(-1,6) #-1을 넣으면 다른 행, 열에 입력한 수에 맞게 알아서 행렬을 나누어준다. 물론 정수 한정
print(Matrix1)