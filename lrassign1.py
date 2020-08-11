import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

wcat = pd.read_csv("calories_consumed.csv")
# print(wcat.columns)
# print(wcat.shape)

# plt.hist(wcat.Calories_Consumed)
# plt.boxplot(wcat.Calories_Consumed,0,"rs",0)
# plt.show()

# plt.hist(wcat.Weight_Gained)
# plt.boxplot(wcat.Weight_Gained,0,"rs",0)
plt.plot(wcat.Calories_Consumed,wcat.Weight_Gained,"bo") #bo specifies blue dot representation
plt.xlabel("Calories_Consumed")
plt.ylabel("Weight_Gained")
# plt.show()
# input()

# print(wcat.Calories_Consumed.corr(wcat.Weight_Gained))
# print(np.corrcoef(wcat.Weight_Gained,wcat.Calories_Consumed)) #correlation is positive, meaning both variables move in the same direction
# input()

import statsmodels.formula.api as smf
model = smf.ols("Weight_Gained~Calories_Consumed",data=wcat).fit()  #R^2 as 89.7% 
# print(model.params)
print(model.summary())
# print(model.conf_int(0.05)) # 95% confidence interval
pred = model.predict(wcat.iloc[:,0]) # Predicted values of weight_gained using the model
print(pred)
# input()

print(pred.corr(wcat.Weight_Gained))
# input()

import matplotlib.pylab as plt
plt.scatter(x=wcat['Calories_Consumed'],y=wcat['Weight_Gained'],color='red')
plt.plot(wcat['Calories_Consumed'],pred,color='black')
plt.xlabel('Calories_Consumed')
plt.ylabel('Weight_Gained')
plt.show()
pred.corr(wcat.Weight_Gained)

rmse = pred-wcat.Weight_Gained

print(np.sqrt(np.mean(rmse*rmse)))





