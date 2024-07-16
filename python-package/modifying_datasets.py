import pandas as pd
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv('learn2clean/datasets/titanic.csv',engine='python',encoding='latin1')
# print(dataset['stag'])
# dataset.sort_values(by=['stag'], inplace=True)
# dataset.to_csv('learn2clean/datasets/turnover.csv', index=False)
# id = 3
# if(len(dataset) > 0):
#     for col in dataset.columns:
#         unique = {}
#         vals = dataset[col].unique()
#         #print(str(vals[0]) + " " + str(type(vals[0])) + " " + str(isinstance(vals[0], (int, float, complex))))
#         if isinstance(vals[0], (int, float, complex)) or str(type(vals[0])) == "<class 'numpy.int64'>": # int is not detected as int that is why there is an or statement 
#             continue
#         for i in  range(len(vals)):
#             unique[vals[i]] = id
#             id += 1
#         for index in dataset.index:
#             temp = dataset[col][index]
#             dataset.at[index, col] = unique.get(temp)


for col in dataset.columns:
    test_val = dataset[col][0]
    #print(dataset[col])
    if isinstance(test_val, (int, float, complex)) or str(type(test_val)) == "<class 'numpy.int64'>": # int is not detected as int that is why there is an or statement 
            continue      
    new_col = LabelEncoder().fit_transform(dataset[col])
    idx = 0
    for index in dataset.index:
          dataset.at[index, col] = new_col[idx]
          idx += 1
    print(new_col)
#print(dataset)

dataset.to_csv("learn2clean/datasets/titanic_edited.csv", index=False)