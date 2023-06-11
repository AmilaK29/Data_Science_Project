# Final Code to be submitted
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings

# Filtering warnings
warnings.filterwarnings('ignore')

# Reading the Employee CSV file
chatterbox = pd.read_csv('employees.csv')

# Droping the duplicate value of the dataset
chatterbox.drop_duplicates()

# Changing the mismatching values in the Name and the title column
titles = ['Ms','Miss','Mr','Mrs']
for index,row in chatterbox.iterrows():
    first_part = row['Name'].split(" ")[0]
    
    if((first_part in titles) and first_part != row['Title']):
        chatterbox.at[index,'Title'] = first_part

# Chaning the mismatching values in the Title column and Gender Columns
for index,row in chatterbox.iterrows():
    
    if(row['Title'] == 'Mr' and row['Gender'] != 'Male'):
        chatterbox.at[index,'Gender'] = 'Male'
        
    elif((row['Title'] == 'Mrs' or row['Title'] == 'Ms') and row['Gender'] != 'Female'):
        chatterbox.at[index,'Gender'] = 'Female'

# Changing the mismatching values in Title column and Marital status column
for index,row in chatterbox.iterrows():
    if(row['Title'] == 'Mrs' and row['Marital_Status'] != 'Married'):
        chatterbox.at[index,'Marital_Status'] = 'Married'
        
# Filling the missing values('0000-00-00') in the Year_of_Birth column using the mode
chatterbox.loc[chatterbox['Year_of_Birth'] == "'0000'",'Year_of_Birth'] = None
chatterbox['Year_of_Birth'].fillna(chatterbox['Year_of_Birth'].mode()[0],inplace = True)

# I have trained a model to fill the missing values in the Marital_Status column

# Getting a copy of the chatterbox data set
chatterbox_copy = chatterbox.copy()
# Converting the Year_of_Birth column to int type
chatterbox_copy['Year_of_Birth'] = chatterbox_copy['Year_of_Birth'].astype(int)

# Seperating dependant variables and independant variables
chatterbox_X = chatterbox_copy[['Title','Gender','Religion_ID','Designation_ID','Status','Year_of_Birth']]
chatterbox_y = chatterbox_copy[['Marital_Status']]

# Getting dummy values for the categorical variables in the chatterbox_X
chatterbox_X = pd.get_dummies(chatterbox_X,drop_first = True)

# Concatenating the encoded values with the dependant variables 
chatterbox_with_dummies = pd.concat([chatterbox_y,chatterbox_X],axis = 1)

# Seperating test set and train set based on the missing values
chatterbox_test = chatterbox_with_dummies[chatterbox_with_dummies['Marital_Status'].isnull()]
chatterbox_train = chatterbox_with_dummies[chatterbox_with_dummies['Marital_Status'].notnull()]

# Dividing the train set into dependant and independant variables
train_X = chatterbox_train[['Religion_ID','Designation_ID','Year_of_Birth','Title_Mr','Title_Mrs','Title_Ms','Gender_Male','Status_Inactive']]
train_Y = chatterbox_train[['Marital_Status']]
train_Y = pd.get_dummies(train_Y,drop_first = True)

# Getting the independant variable for test set.
test_X = chatterbox_test[['Religion_ID','Designation_ID','Year_of_Birth','Title_Mr','Title_Mrs','Title_Ms','Gender_Male','Status_Inactive']]

# Here I have choosed the RandomForestClassifier as it performs well
final_model = RandomForestClassifier(n_estimators = 5)
score = cross_val_score(final_model,train_X,train_Y,cv = 10)


# Training the final model
final_model.fit(train_X,train_Y)
y_pred = final_model.predict(test_X)

# Filling the predicted value
missing_values = chatterbox['Marital_Status'].isnull()
chatterbox.loc[missing_values,'Marital_Status'] = y_pred

# Transforming dates in the string format to date objects
chatterbox['Date_Joined'] = pd.to_datetime(chatterbox['Date_Joined'])

preprocessed_dataset = chatterbox

name = 'employee_preprocess_200196G.csv'

preprocessed_dataset.to_csv(name, index=False)
