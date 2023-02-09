import pandas as pd


df = pd.read_csv('1. NGO.csv', encoding='CP949')
df = df[['CHURN','CONTACT_ID','PLED_NUM_R_CS_INT','PLED_NUM_R_CS_DOM','PLED_NUM_R_CS_DOM','PLED_NUM_R_CS_DOM',
'PLED_NUM_R_PN_DOM','PLED_NUM_R_PN_NKOR','PLED_NUM_R_PN_ALL']]
df = df.dropna()
df = df[df['CHURN']==0]

df['해외아동'] = ''
df['국내아동'] = ''
df['해외사업'] = ''
df['국내사업'] = ''
df['긴급구호사업'] = ''
df['북한사업'] = ''
df['전체사업'] = ''
pled_list_all = ['PLED_NUM_R_CS_INT','PLED_NUM_R_CS_DOM','PLED_NUM_R_CS_DOM','PLED_NUM_R_CS_DOM',
'PLED_NUM_R_PN_DOM','PLED_NUM_R_PN_NKOR','PLED_NUM_R_PN_ALL']
pled_list = ['해외아동','국내아동','해외사업','국내사업','긴급구호사업','북한사업','전체사업']


for j in range(0,len(pled_list),1) : 
    df[pled_list[j]] = df[pled_list_all[j]]
# df.to_csv('협업필터링df2.csv', encoding='utf-8-sig')
# print(df.shape)
# print(df.head())

data = {'uid':[],'후원종류':[],'플릿지수':[]}
data = pd.DataFrame(data)

i = 0
j = 0

for i in range(0,len(df),1) : 
     for j in range(0, len(pled_list),1): 
        # if df[pled_list[j]].iloc[i]>0 : 
            newdata = {'uid': df['CONTACT_ID'].iloc[i],'후원종류': pled_list[j], '플릿지수':df[pled_list[j]].iloc[i]}
            data = data.append(newdata, ignore_index=True)
print(data.shape)
# data.to_csv('협업필터링data2.csv', encoding='utf-8-sig')
# data.to_csv('협업필터링data3.csv', encoding='utf-8-sig')