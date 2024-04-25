import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os
import time

print("program is running")
print()
start_time = time.time() 

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

 
#Label encoding for the categorical features

df['MARITALSTATUS'].unique()
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()



#LABEL ENCODE : EDUCATION

#Ordinal feature---education
#SSC               :1  
#12th              :2  
#Graduate          :3  
#Under Graduate    :3  
#Post Graduate     :4 
#Others            :1   has to be verified by the business end user
#Professional      :3


df.loc[df['EDUCATION'] == 'SSC', ['EDUCATION']]                = 1
df.loc[df['EDUCATION'] == '12TH', ['EDUCATION']]               = 2
df.loc[df['EDUCATION'] == 'GRADUATE', ['EDUCATION']]           = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE', ['EDUCATION']]     = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE', ['EDUCATION']]      = 4
df.loc[df['EDUCATION'] == 'OTHERS', ['EDUCATION']]             = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL', ['EDUCATION']]       = 3

df['EDUCATION'].value_counts()
df['EDUCATION'] = df['EDUCATION'].astype(int)
df.info()


df_encoded = pd.get_dummies(df, columns = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])



#---------------------MACHINE LEARNING MODEL FITTING--------------------------------------------------#

#RANDOM FOREST

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)

rf_classifier = RandomForestClassifier(n_estimators= 200, random_state=42)

rf_classifier.fit(x_train, y_train)

y_pred = rf_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Accuracy : {accuracy}')
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)


for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}")
    print(f"Precision: {precision[i]}")
    print(f"Recall : {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()



#XG BOOST

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

xgb_classifier = xgb.XGBClassifier(objective = "multi:softmax", num_class = 4)

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis = 1)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y_encoded, test_size= 0.2, random_state= 42)

xgb_classifier.fit(x_train, y_train)

y_pred =  xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Accuracy : {accuracy}')
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}")
    print(f"Precision: {precision[i]}")
    print(f"Recall : {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()



#DECISION TREE

from sklearn.tree import DecisionTreeClassifier


y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x,y_encoded, test_size= 0.2, random_state= 42)

dt_model = DecisionTreeClassifier(max_depth = 20, min_samples_split= 10)
dt_model.fit(x_train, y_train)
y_pred = dt_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Accuracy : {accuracy}')
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}")
    print(f"Precision: {precision[i]}")
    print(f"Recall : {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()



#Checking whether P1, P2, P3, P4 is balanced or imbalanced

df_encoded['Approved_Flag'].value_counts()  #balanced

#as approved flag is balanced, so we can focus on accuracy


#------------------Hyperparameter tuning for XGBOOST---------------------------


#Define the hyperparameter grid - 720 combinations

param_grid = {
    'colsample_bytree' : [0.1, 0.3, 0.5, 0.7, 0.9],   #har combination ke saath ekk xgboost ka model
    'learning_rate' : [0.001, 0.01, 0.1, 1],
    'max_depth' : [3,5,8,10],
    'alpha' : [1, 10, 100],
    'n_estimators' : [10, 50, 100]
    }

index = 0

answer_grid = {                                 #har xgboost ke answer ekk dictionary mein 
    'combination' :         [],            
    'train_Accuracy' :      [],
    'test_accuracy':        [],
    'colsample_bytree' :    [],
    'learning_rate':        [],
    "max_depth":            [],
    'alpha':                [],
    'n_estimators':         []
    }

#Loop throught each combination of hyperparameters

for colsample_bytree in param_grid['colsample_bytree']:
    for learning_rate in param_grid['learning_rate']:
        for max_depth in param_grid['max_depth']:
            for alpha in param_grid['alpha']:
                for n_estimators in param_grid['n_estimators']:
                    
                    index = index + 1
                    #define the train XGBOOST model
                    
                    model = xgb.XGBClassifier(objective= "multi:softmax", 
                                              num_class = 4,
                                              colsample_bytree = colsample_bytree,
                                              learning_rate = learning_rate,
                                              max_depth = max_depth, 
                                              alpha = alpha,
                                              n_estimators = n_estimators)
                    
                    y = df_encoded['Approved_Flag']
                    x = df_encoded.drop(['Approved_Flag'], axis = 1)
                    
                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y)
                    
                    x_train, x_test, y_train, y_teet = train_test_split(x, y_encoded, test_size= 0.2,random_state=2 )
                    
                    model.fit(x_train, y_train)
                    
                    
                    #Predict on training and testing sets
                    
                    y_pred_train = model.predict(x_train)
                    y_pred_test = model.predict(x_test)
                    
                    #Calculate train and test results
                    
                    train_accuracy = accuracy_score(y_train, y_pred_train)
                    test_accuracy  = accuracy_score(y_test, y_pred_test)
                    
                    
                    #Include into the lists
                    answer_grid['combination'].append(index)
                    answer_grid['train_Accuracy'].append(train_accuracy)
                    answer_grid['test_accuracy'].append(test_accuracy)
                    answer_grid['colsample_bytree'].append(colsample_bytree)
                    answer_grid['learning_rate'].append(learning_rate)
                    answer_grid['max_depth'].append(max_depth)
                    answer_grid['alpha'].append(alpha)
                    answer_grid['n_estimators'].append(n_estimators)
                    
                    
                    #Print results for this combination
                    
                    print(f"Combination{index}")
                    print(f"Colsample_bytree {colsample_bytree}, learning_rate : {learning_rate}, max_depth : {max_depth}, alpha : {alpha}, n_estimators : {n_estimators}")
                    print(f"Train Accuracy : {train_accuracy : .2f}")
                    print(f"Test Accuracy : {test_accuracy : .2f}")
                    print("-" * 30)


#Drawback - Xgboost in itself is a heavy algo, so so many combinations lead to computational inefficiency



#-------------------------------------Finally fitting the model with the best parameters--------------------------------------------------

a3 = pd.read_excel("C:/Users/DELL/OneDrive/Desktop/Desktop/College/DATA SCIENCE/PROJECTS/Credit Risk Modelling/Unseen_Dataset.xlsx")
a3
cols_in_df = list(a3.columns)

df_unseen = a3[cols_in_df]
cols_in_df
df.columns

df_unseen.loc[df_unseen['EDUCATION'] == 'SSC', ['EDUCATION']]                = 1
df_unseen.loc[df_unseen['EDUCATION'] == '12TH', ['EDUCATION']]               = 2
df_unseen.loc[df_unseen['EDUCATION'] == 'GRADUATE', ['EDUCATION']]           = 3
df_unseen.loc[df_unseen['EDUCATION'] == 'UNDER GRADUATE', ['EDUCATION']]     = 3
df_unseen.loc[df_unseen['EDUCATION'] == 'POST-GRADUATE', ['EDUCATION']]      = 4
df_unseen.loc[df_unseen['EDUCATION'] == 'OTHERS', ['EDUCATION']]             = 1
df_unseen.loc[df_unseen['EDUCATION'] == 'PROFESSIONAL', ['EDUCATION']]       = 3

df_unseen['EDUCATION'].value_counts()
df_unseen['EDUCATION'] = df_unseen['EDUCATION'].astype(int)
df_unseen.info()

df_encoded_unseen = pd.get_dummies(df_unseen, columns =  ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])

model = xgb.XGBClassifier(objective= 'multi:softmax',
                          num_class = 4, 
                          colsample_bytree = 0,
                          learning_rate = 1,
                          max_depth = 3,
                          alpha = 10,
                          n_estimators = 100)

model.fit(x_train, y_train)

y_pred_unseen = model.predict(df_encoded_unseen)

a3['Target_variable'] = y_pred_unseen


a3.to_excel("Final predictions.xlsx", index = False)








