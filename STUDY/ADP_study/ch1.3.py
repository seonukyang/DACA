import pandas as pd
import numpy as np

#단일값 추출
array1 = np.arange(start=1, stop=10)

value = array1[2]
print('value : ',value)
print('value type : ',type(value))

list1 = [1,2,3,4,5]
print(list1[1])

array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3,3)
print(array2d)
print('(row=0,col=0) index 가리키는 값 : ',array2d[0,0])
print('(row=0,col=1) index 가리키는 값 : ',array2d[0,1])
print('(row=1,col=0) index 가리키는 값 : ',array2d[1,0])
print('(row=2,col=2) index 가리키는 값 : ',array2d[2,2])

array3 = array1[0:3]

list2 = [[1,2,3],[4,5,6],[7,8,9]]
print([list2[0][0:2],list2[1][0:2]])

array1d = np.arange(1,10,1)
array2d = array1d.reshape(3,3)

array3 = array2d[[0,1],2]
print('array2d[[0,1],2] => ',array3.tolist())

array4 = array2d[[0,1], 0:2]
print('array2d[[0,1], 0:2] => ',array4.tolist())

array5 = array2d[[0,1]]
print('array2d[[0,1]]=> ', array5.tolist())

array1d = np.arange(1,10,1)
array3 = array1d[array1d > 5]
print('array1d > 5 불링 인덱싱 결과 값 : ',array3)
print(array1d>5)

boolean_indexes = np.array(array1d>5)
array3 = array1d[boolean_indexes]
print('불링 인덱스로 필터링 결과 : ',array3)


#행렬의 정렬
org_array = np.array([3,1,9,5])
print('원본 행렬 : ',org_array)
sort_array1 = np.sort(org_array)
print('np.sort()호출 후 반환된 정렬 행렬 : ',sort_array1)
print('np.sort()호출 후 원본 행렬 : ',org_array)
sort_array2 = org_array.sort()
print('org_array.sort() 호출 후 반환된 행렬 : ',sort_array2)
print('org_array.sort() 호출 후 원본 행렬:',org_array)

sort_array1_desc = np.sort(org_array)[::-1]

array2d = np.array([[8,12],[7,1]])
sort_array2d_axis0 = np.sort(array2d, axis=0)
print('로우 방향으로 정렬:\n', sort_array2d_axis0)

sort_array2d_axis1 = np.sort(array2d, axis=1)
print('칼럼 방향으로 정렬:\n', sort_array2d_axis1)

#정렬된 행렬의 인덱스를 반환하기
org_array = np.array([3,1,9,5])
sort_indices = np.argsort(org_array)
print(type(sort_indices))
print('행렬 정렬 시 원본 행렬의 인덱스:', sort_indices)

org_array = np.array([3,1,9,5])
sort_indices_desc = np.argsort(org_array)[::-1]
print('행렬 내림차순 정렬 시 원본 행렬의 인덱스 : ',sort_indices_desc)

name_array = np.array(['John','Mike','Sarah','Kate','Samuel'])
score_array = np.array([78,95,84,98,88])

sort_indices_asc = np.argsort(score_array)
print('성적 오름차순 정렬 시 score_array의 인덱스 : ', sort_indices_asc)
#정렬된 순서로 인덱스를 뽑아서 이름 리스트에 집어넣으면 그 순서대로 출력이 된다.
print('성적 오름차순으로 name_array의 이름 출력:', name_array[sort_indices_asc])


#선형대수 연산 - 행렬 내적과 전치 행렬 구하기
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[7,8],[9,10],[11,12]])
dot_product = np.dot(A,B)
print('행렬 내적 결과 \n',dot_product)

#전치 행렬
A = np.array([[1,2],[3,4]])
transpose_mat = np.transpose(A)
print('A의 전치 행렬\n',transpose_mat)


