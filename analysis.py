import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os

a1 = pd.read_excel("C:/Users/DELL/OneDrive/Desktop/Desktop/College/DATA SCIENCE/PROJECTS/Credit Risk Modelling/DATA/case_study1.xlsx")
a2 = pd.read_excel("C:/Users/DELL/OneDrive/Desktop/Desktop/College/DATA SCIENCE/PROJECTS/Credit Risk Modelling/DATA/case_study2.xlsx")

print("df2 info: ",a1.info())
print("df2 info: ",a2.info())

df1 = a1.copy()
df2 = a2.copy()

df1.info()


#Remove nulls: as only 44 rows have -99999 values, we can remove the whole row
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

df1.shape

#removing null values for df2. for each columns in df2 if null values> 10000, we will drop the columns
#columns
l = []

for i in df2.columns: 
    if df2.loc[df2[i] == -99999].shape[0]>10000:   #if null values is greater than 10000
        l.append(i)
df2 = df2.drop(columns = l, axis = 1)


for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]

#checking the null values

df1.isna().sum()
df2.isna().sum()


#merging df1 and df2

df = pd.merge(df1, df2, how = "inner", left_on = ['PROSPECTID'], right_on = ['PROSPECTID'])


#finding categorical variables

for i in df.columns:
    if df[i].dtype == 'O':
        print(i)
        
df['MARITALSTATUS'].value_counts()



#chi square test: are the categorical variables and target variable (approved flag) associated?

#H0 : The categorical variable is not associated 
#H1 : The cateogircal variable is associated with the target variable

for i in ['MARITALSTATUS','EDUCATION', 'GENDER', 'last_prod_enq2' ,'first_prod_enq2']:
    chi2, pval,_,_ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i, '---', pval)

#Note: e mean (10 to the power)
#Interpretation: Since all the variables have p values less than 0.05, we can reject H0, hence accept al variables




#NUMERICAL COLUMNS

numerical_columns = []
for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID', 'Approved_Flag']:
        numerical_columns.append(i)




#CHECKING FOR MULTICOLIINEARITY - VIF

vif_data = df[numerical_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range (0, total_columns): #saare 72 columns mein jaa rahe hai
    
    vif_value = variance_inflation_factor(vif_data, column_index)  #calculate vif
    print(column_index, '-----', vif_value) #prinitng vif
    
    if vif_value <=6:
        columns_to_be_kept.append(numerical_columns[i]) #columns rakh rahe hai if less than 6
        column_index = column_index + 1
        
    else: 
        vif_data = vif_data.drop([numerical_columns[i]], axis =1)  #greater than 6 hai toh drop kar rahe hai
    
    
    
#initally 72 numerical variables the, ab 40 features ho gaye



#ANOVA- checking association 


from scipy.stats import f_oneway

columns_to_be_kept_numerical = []

for i in columns_to_be_kept:  #40 columns to be kept
    a = list(df[i])
    b = list(df['Approved_Flag'])
    
    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']


    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)



#now we have 37 variables
























