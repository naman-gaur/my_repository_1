import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

wcat = pd.read_csv("delivery_time.csv")
# print(wcat.columns)
# print(wcat.shape)

# plt.hist(wcat.Sorting_time)
# plt.boxplot(wcat.Sorting_time,0,"rs",0)
# plt.show()

# plt.hist(wcat.Delivery_time)
# plt.boxplot(wcat.Delivery_time,0,"rs",0)
plt.plot(wcat.Sorting_time,wcat.Delivery_time,"bo") #bo specifies blue dot representation
plt.xlabel("Sorting_time")
plt.ylabel("Delivery_time")
# plt.show()

print(wcat.Sorting_time.corr(wcat.Delivery_time))
print(np.corrcoef(wcat.Delivery_time,wcat.Sorting_time)) #correlation is positive, meaning both variables move in the same direction
input()

import statsmodels.formula.api as smf
model = smf.ols("Delivery_time~Sorting_time",data=wcat).fit()  #R^2 as 68.2
# print(model.params)
print(model.summary())

# print(model.conf_int(0.05)) # 95% confidence interval

pred = model.predict(wcat.iloc[:,0]) # Predicted values of Delivery_time using the model
print(pred)
# input()

pred.corr(wcat.Delivery_time)

import matplotlib.pylab as plt
plt.scatter(x=wcat['Sorting_time'],y=wcat['Delivery_time'],color='red')
plt.plot(wcat['Sorting_time'],pred,color='black')
plt.xlabel('Sorting_time')
plt.ylabel('Delivery_time')
plt.show()
pred.corr(wcat.Delivery_time)

rmse = pred-wcat.Delivery_time

print(np.sqrt(np.mean(rmse*rmse)))