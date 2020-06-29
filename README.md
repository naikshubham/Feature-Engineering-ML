# Feature-Engineering-ML
Feature Engineering techniques for ML

### Different types of data
1. Continuos : either integers(or whole numbers) or floats(decimals)
2. Categorical : one of a limited set of values e.g gender, country
3. Ordinal : ranked values, often with no detail of distance between them
4. Boolean : True/False
5. Datetime : dates and times

#### Selecting specific data types

```python
only_ints = df.select_dtypes(include=['int'])
print(only_ints.column)
```

### Categorical variables
- Represents groups that are qualitative in nature.
- We need to encode them as numeric values to use them in ML models. Assigning each category with number can lead to errors.
- These categories are unordered, so assigning this order may greatly penalize the effectiveness of our model. Assiginig number would imply some form of ordering to the categories.
- Instead we can use techniques like one-hot encoding, in doing so our model can leverage the information of what country is given, without inferring any order between the different options.

#### Encoding categorical features
1. One-hot encoding : by default, pandas performs one-hot encoding when we use the get_dummies() function.
- One hot encoding converts n_categories into n_features. We can use get_dummies to one-hot encode columns. The function takes a Dataframe and a list of categorical columns to be converted.

```python
pd.get_dummies(df, columns=['Country'], prefix = 'C')
```

2. Dummy encoding

```python
pd.get_dummies(df, columns=['Country'], drop_first=True, prefix='C')
```

- Dummy encoding creates n-1 features for n categories, omitting the first category.

### Numeric variables

#### Types of numeric features
- Age
- Price
- Counts
- Geospatial data


- Depending on the usecase numeric features can be treated in several different ways.

#### Binarizing numeric variables
- Variable with value 0 is 0 and any other value is 1.

```python
df['Binary_Violation'] = 0
df.loc[df['Number_of_Violations'] > 0, 'Binary_Violation'] = 1
```

#### Binning numeric varaibles
- Often useful for features like age, wage brackets etc.

```python
import numpy as np
df['Binned_Group'] = pd.cut(df['Number_of_Violations'],
                    bins=[-np.inf, 0, 2, np.inf],
                    labels = [1,2,3]
                    )
```























