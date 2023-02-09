point = 20
if point >= 30:
    print('vip')
else:
    print("member")


for a in [1,2,3,4,5]:
    print(a)

for b in range(1,10,2):
    print(b)

#while
#초기화
customer_count = 0
sum1 = 0

#조건검사
while(sum1<20):
    customer_count = customer_count + 1
    sum1 = sum1 + customer_count
    print("customer_count = ",customer_count, "sum1=",sum1)