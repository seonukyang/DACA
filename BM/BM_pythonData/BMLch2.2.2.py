#if 문
point = 20
age = 65
if point >=60 and age >= 60:
    print('vip')
else :
    print('member')

#for문
for a in [1,2,3,4,5] : 
    print(a)

for b in range(1,10,2):
    print(b)

# while 문
#1. 초기화
customer_count = 0
sum1 = 0
print('customer_count sum')

#2 조건 검사
while (sum1 < 20) : 
    customer_count = customer_count + 1
    sum1 = sum1 + customer_count
    print(customer_count, sum1)
