import importlib  
import learn2clean.qlearning.qlearner as ql
import learn2clean.loading.reader as rd 
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
import os 
import sys
import pandas as pd
# qlearner_class = importlib.import_module("python-package/learn2clean/qlearning")
# print(qlearner_class)
#print(os.getcwd())
#print(sys.path)
# titanic = ["learn2clean/datasets/titanic/titanic_train.csv","learn2clean/datasets/titanic/test.csv"]
# hr = rd.Reader(sep=',', verbose=False, encoding=False) 
# dataset = hr.train_test_split(titanic, 'Survived')
#dataset = pd.read_csv('learn2clean/datasets/subset.csv')




x = pd.read_csv('learn2clean/datasets/S1Data.csv')
y = x[["time", "default_time"]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(f"-----------------------STARTING TEST----------------------\n\n\n\n")
# print(f'X_TRAIN: {X_train} \n \n \n')
# print(f'X_TEST: {X_test} \n \n \n')
# print(f'Y_TRAIN: {y_train} \n \n \n')
# print(f'Y_TEST: {y_test} \n \n \n')
test_dataset = {"train": X_train, "test": X_test, "target":y_train, "target_test": y_test}





event_name = 'status'
time_name = 'age'

dataset_path = ['learn2clean/datasets/gbsg.csv']
reader = rd.Reader(sep=',',verbose=False, encoding=False)
dataset = reader.train_test_split(dataset_path, target_name=event_name, duration_column=time_name)
#print(f'\n \n \n{dataset}')
# print(dataset['train'])
# print(dataset['test'])
# print(dataset['target'])
# print(dataset['target_test'])
#dataset['train'].drop(['novator', 'independ', 'extraversion', 'greywage', 'head_gender', 'coach', 'traffic', 'industry'], axis=1, inplace=True)
#dataset['test'].drop(['novator', 'independ', 'extraversion', 'greywage', 'head_gender', 'coach', 'traffic', 'industry'], axis=1, inplace=True)




l2c = ql.Qlearner(dataset=dataset, goal='COX', target_goal=event_name, target_prepare=None, time_column=time_name, file_name='test_dataset', threshold=0.6)
l2c.learn2clean()

'''
output for subset.csv with all columns considered:

**** Best strategy ****
('test_dataset', 'learn2clean', 'COX', 'default_time', None, 'ZSB -> MEAN -> COX', 'C-index', 0.9983189960130536, 473.9223654270172)


output for subset.csv with all columns considered and manual calculation of c-index:
**** Best strategy ****
('test_dataset', 'learn2clean', 'COX', 'default_time', None, 'MF -> COX', 'C-index', 0.6473760483380622, 1630.359694480896)


output for subset.csv with columns ['payoff_time', 'status_time'] dropped:

**** Best strategy ****
('test_dataset', 'learn2clean', 'COX', 'default_time', None, 'ZSB -> MEAN -> COX', 'C-index', 0.8214640160916645, 408.4712088108063)


-------------------------------------------------------------
output for turnover_edited.csv:

**** Best strategy ****
('test_dataset', 'learn2clean', 'COX', 'event', None, 'MM -> ED -> COX', 'C-index', 0.6146628411741013, 12.01918888092041)


output for turnover_edited with some columns dropped:

**** Best strategy ****
('test_dataset', 'learn2clean', 'COX', 'event', None, 'MM -> ED -> COX', 'C-index', 0.5840712918855095, 9.686051607131958)


-------------------------------------------------------------
output for gbsg.csv:

**** Best strategy ****
('test_dataset', 'learn2clean', 'COX', 'status', None, 'IQR -> COX', 'C-index', 0.7521206409048068, 3.659566640853882)
'''


