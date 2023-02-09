import pandas as pd
import pingouin as pg
df = pd.read_csv('Ashopping.csv', encoding='CP949')
X = df[['매장_청결성','공간_편의성','시야_확보성','음향_적절성','상품_다양성']]
X1 = df[['공간_편의성','시야_확보성','음향_적절성','상품_다양성']]
X2 = df[['매장_청결성','시야_확보성','음향_적절성','상품_다양성']]
X3 = df[['매장_청결성','공간_편의성','음향_적절성','상품_다양성']]
X4 = df[['매장_청결성','공간_편의성','시야_확보성','상품_다양성']]
X5 = df[['매장_청결성','공간_편의성','시야_확보성','음향_적절성']]
ca_X = pg.cronbach_alpha(data=X)
ca_X1 = pg.cronbach_alpha(data=X1)
ca_X2 = pg.cronbach_alpha(data=X2)
ca_X3 = pg.cronbach_alpha(data=X3)
ca_X4 = pg.cronbach_alpha(data=X4)
ca_X5 = pg.cronbach_alpha(data=X5)
print('X : ', ca_X)
print('X1 : ', ca_X1)
print('X2 : ', ca_X2)
print('X3 : ', ca_X3)
print('X4 : ', ca_X4)
print('X5 : ', ca_X5)