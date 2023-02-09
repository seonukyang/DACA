df_X['연령대'] = ''
for i in range(0,len(df_X),1) : 
    for j in range(0, len(df),1):
        if df_X['CONTACT_ID'].iloc[i] == df['CONTACT_ID'].iloc[j] : 
            if df['AGE'][j] == 0 :
                df_X['연령대'].iloc[i] = -1
            elif df['AGE'][j] < 10 :
                df_X['연령대'].iloc[i] = 0   
            elif df['AGE'][j] < 20 :
                df_X['연령대'].iloc[i] = 1
            elif df['AGE'][j] < 30 :
                df_X['연령대'].iloc[i] = 2
            elif df['AGE'][j] < 40 :
                df_X['연령대'].iloc[i] = 3
            elif df['AGE'][j] < 50 :
                df_X['연령대'].iloc[i] = 4
            elif df['AGE'][j] < 60 :
                df_X['연령대'].iloc[i] = 5
            elif df['AGE'][j] < 70 :
                df_X['연령대'].iloc[i] = 6
            elif df['AGE'][j] < 80 :
                df_X['연령대'].iloc[i] = 7
            else : df_X['연령대'].iloc[i] = 8
df_X = df_X[df_X['연령대']>=0]
df_X.index = range(0,len(df_X),1)
#등분산 검정
df1 = df_X[['PAY_SUM_PAYMENTAMOUNT','고객등급','연령대']]
고객등급 = []
for i in range(1,6,1):
    고객등급.append(df1[df1.고객등급==i].PAY_SUM_PAYMENTAMOUNT)
print('고객등급x총금액',sp.stats.levene(고객등급[0],고객등급[1],고객등급[2],고객등급[3],고객등급[4]))

연령대 = []
for i in range(0,9,1):
    연령대.append(df1[df1.연령대==i].PAY_SUM_PAYMENTAMOUNT)
print('연령대x총금액', sp.stats.levene(연령대[0],연령대[1],연령대[2],연령대[3],연령대[4],연령대[5],연령대[6],연령대[7],연령대[8]))

#이원분산분석
print('이원분산분석\n', pg.anova(dv='PAY_SUM_PAYMENTAMOUNT', between=['고객등급','연령대'], data=df_X))

#사후분석
df1['고객등급'] = df1['고객등급'].astype(str)
df1['연령대'] = df1['연령대'].astype(str)
print('고객등급x총매출액\n', scikit_posthocs.posthoc_scheffe(df1, val_col='PAY_SUM_PAYMENTAMOUNT', group_col='고객등급'))
print('연령대x총매출액\n', scikit_posthocs.posthoc_scheffe(df1, val_col='PAY_SUM_PAYMENTAMOUNT', group_col='연령대'))

#고객등급, 연령대별 평균 총 매출액
pd.pivot_table(df1, index='고객등급', columns='연령대', values='PAY_SUM_PAYMENTAMOUNT', aggfunc=np.mean)