import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

wcat = pd.read_csv("Salary_data.csv")
# print(wcat.columns)
# print(wcat.shape)

# plt.hist(wcat.YearsExperience)
# plt.boxplot(wcat.YearsExperience,0,"rs",0)
# plt.show()

# plt.hist(wcat.Salary)
# plt.boxplot(wcat.Salary,0,"rs",0)
plt.plot(wcat.YearsExperience,wcat.Salary,"bo") #bo specifies blue dot representation
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
# plt.show()
# input()

print(wcat.YearsExperience.corr(wcat.Salary))
print(np.corrcoef(wcat.Salary,wcat.YearsExperience)) #correlation is positive, meaning both variables move in the same direction
# input()

import statsmodels.formula.api as smf
model = smf.ols("Salary~YearsExperience",data=wcat).fit()  #R^2 as 95.7%
# print(model.params)
print(model.summary())
# print(model.conf_int(0.05)) # 95% confidence interval
pred = model.predict(wcat.iloc[:,0]) # Predicted values of weight_gained using the model
print(pred)
# input()

print(pred.corr(wcat.Salary))
# input()

import matplotlib.pylab as plt
plt.scatter(x=wcat['YearsExperience'],y=wcat['Salary'],color='red')
plt.plot(wcat['YearsExperience'],pred,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
pred.corr(wcat.Salary)

rmse = pred-wcat.Salary

print(np.sqrt(np.mean(rmse*rmse)))





