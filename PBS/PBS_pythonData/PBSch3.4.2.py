import numpy as np
a1 = np.arange(1,10,2) #시작, 끝, 간격
print(a1)
a2=np.arange(10) #0부터 열번째까지
print(a2)

arr1 = np.array([10,15,20,30])
arr2 = np.array([1,1,2,3])
print(arr1+arr2)
print(arr1-arr2)
print(arr1*arr2)
print(arr1/arr2)

b1 = np.array([0,10,20,30,40,50])
b1
print(b1[4])
b1[5]=70
print(b1)
print(b1[[1,3,4]])

c1 = np.array([0,1,2,3,4,5])
print(c1[1:4]) #인댁스 1번부터 4번 이전까지 즉 1,2,3
print(c1[2:]) #인댁스 2번부터 끝까지
c1[2:5] = np.array([25,35,45]) #2:5면 2,3,4를 의미
print(c1)