import pandas as pd


df = pd.read_csv('1. NGO.csv', encoding='CP949')
df = df[['CHURN','CONTACT_ID','PAY_SUM_PAYMENT_R_CS_INT','PAY_SUM_PAYMENT_R_CS_DOM','PAY_SUM_PAYMENT_R_PN_INT','PAY_SUM_PAYMENT_R_PN_FST',
'PAY_SUM_PAYMENT_R_PN_DOM','PAY_SUM_PAYMENT_R_PN_NKOR','PAY_SUM_PAYMENT_R_PN_ALL','PAY_SUM_PAYMENT_O_GN_INT','PAY_SUM_PAYMENT_O_GN_DOM',
'PAY_SUM_PAYMENT_O_PN_INT','PAY_SUM_PAYMENT_O_PN_FST','PAY_SUM_PAYMENT_O_PN_DOM','PAY_SUM_PAYMENT_O_PN_NKOR','PAY_SUM_PAYMENT_O_PN_ALL']]
df = df.dropna()
df = df[df['CHURN']==0]

df['해외아동'] = ''
df['국내아동'] = ''
df['해외사업'] = ''
df['국내사업'] = ''
df['긴급구호사업'] = ''
df['북한사업'] = ''
df['전체사업'] = ''
pled_list_all = ['PAY_SUM_PAYMENT_R_CS_INT','PAY_SUM_PAYMENT_R_CS_DOM','PAY_SUM_PAYMENT_R_PN_INT','PAY_SUM_PAYMENT_R_PN_FST',
'PAY_SUM_PAYMENT_R_PN_DOM','PAY_SUM_PAYMENT_R_PN_NKOR','PAY_SUM_PAYMENT_R_PN_ALL','PAY_SUM_PAYMENT_O_GN_INT','PAY_SUM_PAYMENT_O_GN_DOM',
'PAY_SUM_PAYMENT_O_PN_INT','PAY_SUM_PAYMENT_O_PN_FST','PAY_SUM_PAYMENT_O_PN_DOM','PAY_SUM_PAYMENT_O_PN_NKOR','PAY_SUM_PAYMENT_O_PN_ALL']
pled_list = ['해외아동','국내아동','해외사업','국내사업','긴급구호사업','북한사업','전체사업']

for i in range(0,len(df),1) : 
    for j in range(0,len(pled_list),1) : 
        df[pled_list[j]].iloc[i] = df[pled_list_all[j]].iloc[i]+df[pled_list_all[j+7]].iloc[i]

df.to_csv('협업필터링df.csv', encoding='utf-8-sig')
# print(df.shape)
# print(df.head())

data = {'uid':[],'후원종류':[],'납입금액':[]}
data = pd.DataFrame(data)

i = 0
j = 0

for i in range(0,len(df),1) : 
     for j in range(0, len(pled_list),1): 
        print(df[pled_list[j]].iloc[i])
        if df[pled_list[j]].iloc[i]>0 : 
            newdata = {'uid': df['CONTACT_ID'].iloc[i],'후원종류': pled_list[j], '납입금액':df[pled_list[j]].iloc[i]}
            data = data.append(newdata, ignore_index=True)
print(data.shape)
# data.to_csv('협업필터링data.csv', encoding='utf-8-sig')
