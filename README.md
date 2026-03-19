## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING:
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from scipy.stats import boxcox
```
Load the Dataset

```python
import pandas as pd

data = pd.read_csv("Data_to_Transform.csv")

print("Original Dataset:")
print(data.head())
```
Handle Missing Values (Fill numeric columns with mean)

```python
data.fillna(data.mean(numeric_only=True), inplace=True)
```
Select a suitable numeric column for transformation

```python
numeric_column = data.select_dtypes(include=np.number).columns[0]

print(f"\nColumn Selected for Transformation: {numeric_column}")
```
Keep only positive values for log and boxcox

```python
positive_data = data[data[numeric_column] > 0].copy()
```

Log Transformation

```python
positive_data['Log_Transform'] = np.log(positive_data[numeric_column])
```
Reciprocal Transformation

```python
positive_data['Reciprocal_Transform'] = 1 / positive_data[numeric_column]
```
Square root Transformation

```python
positive_data['Sqrt_Transform'] = np.sqrt(positive_data[numeric_column])
```
Square Transform

```python
positive_data['Square_Transform'] = np.square(positive_data[numeric_column])
```
Box_cox Transform

```python
positive_data['BoxCox_Transform'], lambda_value = boxcox(positive_data[numeric_column])

print(f"\nBox-Cox Lambda Value: {lambda_value}")
```
Power Transform

```python
pt = PowerTransformer(method='yeo-johnson')
data['YeoJohnson_Transform'] = pt.fit_transform(data[[numeric_column]])
```
StandardScaler

```python
scaler = StandardScaler()
data['Standard_Scaled'] = scaler.fit_transform(data[[numeric_column]])
```
Save the transformed dataset

```python
positive_data.to_csv('Transformed_Positive_Data.csv', index=False)
data.to_csv('Transformed_Full_Data.csv', index=False)

print("\nTransformation Completed Successfully.")
print("\nTransformed Dataset Preview:")
print(positive_data.head())
```

# Output:

<img width="819" height="380" alt="Screenshot 2026-03-10 091725" src="https://github.com/user-attachments/assets/f9863303-2dc0-4e32-bfd5-90523f7ebefd" />


<img width="1006" height="478" alt="Screenshot 2026-03-10 091757" src="https://github.com/user-attachments/assets/9370443b-f877-41d2-a660-1af2f2c8ca73" />


<img width="1017" height="203" alt="Screenshot 2026-03-10 091849" src="https://github.com/user-attachments/assets/e0b4fc56-e91a-48b0-a880-b389c6d1b3fa" />


<img width="873" height="598" alt="Screenshot 2026-03-10 091922" src="https://github.com/user-attachments/assets/8a07e6f5-48f7-421b-bded-401538c67040" />





      
# RESULT:
Thus the Implementation of Feature Encoding and Feature Transformation executed successfully.

       
