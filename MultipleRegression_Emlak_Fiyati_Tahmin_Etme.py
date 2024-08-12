#Gerekli kütüphanelerimizi içe aktarıyoruz.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


#Csv dosyasındaki veriyi pandas kütüphanesi ile dataframe'e çeviriyoruz.
df = pd.read_csv("multilinearregression.csv", sep=";")
#print(df.head())


#Linear Regression modelini tanımlıyoruz.
reg = linear_model.LinearRegression()
reg.fit(df[["alan", "odasayisi", "binayasi"]].values, df["fiyat"].values)


#Dışarıdan test amaçlı verimizi yapay zekaya soruyoruz.
#print(reg.predict([[230, 4, 10]]))
#print(reg.predict([[230, 6, 0]]))
#print(reg.predict([[355, 3, 20]]))

#print("-----------------------------")


#Dışarıdan verilen test verilerini tek tek yazmak zorunda değiliz.
#Burada kullandığımız array içerisinde bütün verileri belirtebiliriz.
#print(reg.predict([[230, 4, 10], [230, 6, 0], [355, 3, 20]]))


#Şimdi yapay zeka algoritmamızın b1, b2 ve b3 katsayılarını görelim.
#print(reg.coef_)


#Algoritmamızın sabit değerini görelim.
#print(reg.intercept_)


#Şimdi biz bu algoritmanın formülünü ve nasıl çalıştığını biliyoruz.
#Bunun sağlamasını yapalım. Formülümüz: y = a + b1X1 + b2X2 + b3X3 + ...
a = reg.intercept_
b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 230
x2 = 4
x3 = 10
y = a + b1*x1 + b2*x2 + b3*x3
print(y)
