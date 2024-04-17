# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
 ```
import pandas as pd
from scipy import stats
import numpy as np
```
 import pandas as pd
df=pd.read_csv("/content/bmi.csv")
df
```
![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/c1d6dbcd-b572-4c42-8622-ccaebcd317cf)

```
df.head()
```
![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/93f54ada-c7c0-4e89-8d32-53b2609e3d11)
```
import numpy as np
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/ba13e144-24b0-4f3c-b009-616a0a5a5a33)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/fa29458c-eedd-495a-ba5d-322120cf9b42)
```
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/1cf43d46-1fe8-4129-a80c-aace8c1f2763)
```
from sklearn.preprocessing import Normalizer
Scaler=Normalizer
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/0047cb2c-2b3a-47ed-9ff2-1b64276673ee)
```
df=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/bd16f86e-1b97-4bb9-ba84-2c545b5c022b)
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data = pd.read_csv("/content/income(1) (1).csv")
data
```
![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/defab84b-528f-4d01-bec0-4eb7e6635c33)
```
data.isnull().sum()
```
![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/f0d421ac-0206-447b-a27a-ab9859579de7)
```
missing = data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/abd61518-0e3d-4798-9f99-ea33f2635169)

```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/39c6e2bf-353f-44d9-af88-423191d5c75f)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/a964b848-188e-427e-82f5-5fab1b9c8503)
```
data2
![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/6e1b254f-4baa-4660-8b87-7bc1864812d5)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/7d864a7c-c1f6-4d78-aa80-1398806a20a1)
```
columns_list=list(new_data.columns)
print(columns_list)
```

![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/f63f436c-be09-41dd-809e-6e31c3219f48)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/c13ff556-835e-4b8c-9b02-dbfd12dd5f5c)
```
y=new_data['SalStat'].values
print(y)
```

![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/dcd6c3ad-bbc0-4a79-ae3c-8aeb0978ff59)

```
x = new_data[features].values
print(x)
```

![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/47fd2a3f-8393-4f6f-9a46-ad52d6dd088d)
```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state = 0)
KNN_classifier = KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x, train_y)
```

![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/12c623fe-67b6-4db8-aca2-2b15f02c3e39)

```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```

![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/30a8a88d-f809-432d-83eb-bcf529a04e15)
```
print( 'Misclassified samples: %d' % (test_y != prediction).sum())
```

![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/af63f87a-7c2e-4997-b629-30458fcf831a)
```
data.shape
```

![image](https://github.com/23007232/EXNO-4-DS/assets/139115574/caddb12e-7af9-4c89-9f55-3a0fdefa2eda)

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
