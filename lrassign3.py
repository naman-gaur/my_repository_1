import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

wcat = pd.read_csv("emp_data.csv")
# print(wcat.columns)
# print(wcat.shape)

# plt.hist(wcat.Salary_hike)
# plt.boxplot(wcat.Salary_hike,0,"rs",0)
# plt.show()

# plt.hist(wcat.Churn_out_rate)
# plt.boxplot(wcat.Churn_out_rate,0,"rs",0)
plt.plot(wcat.Salary_hike,wcat.Churn_out_rate,"bo") #bo specifies blue dot representation
plt.xlabel("Salary_hike")
plt.ylabel("Churn_out_rate")
# plt.show()
# input()

print(wcat.Salary_hike.corr(wcat.Churn_out_rate))
print(np.corrcoef(wcat.Churn_out_rate,wcat.Salary_hike)) #correlation is negative, meaning that when one variable's value increases, the other variables' values decrease
# input()

import statsmodels.formula.api as smf
model = smf.ols("Churn_out_rate~Salary_hike",data=wcat).fit()  #R^2 as 83.1% 
# print(model.params)
print(model.summary())
# print(model.conf_int(0.05)) # 95% confidence interval
pred = model.predict(wcat.iloc[:,0]) # Predicted values of weight_gained using the model
print(pred)
# input()

print(pred.corr(wcat.Churn_out_rate))
# input()

import matplotlib.pylab as plt
plt.scatter(x=wcat['Salary_hike'],y=wcat['Churn_out_rate'],color='red')
plt.plot(wcat['Salary_hike'],pred,color='black')
plt.xlabel('Salary_hike')
plt.ylabel('Churn_out_rate')
plt.show()
pred.corr(wcat.Churn_out_rate)

rmse = pred-wcat.Churn_out_rate

print(np.sqrt(np.mean(rmse*rmse)))





