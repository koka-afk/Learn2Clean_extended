from learn2clean.qlearning import qlearner as ql
import learn2clean.loading.reader as rd 
from learn2clean.qlearning import survival_qlearner as survival_ql
import pandas as pd
from sksurv.datasets import load_gbsg2

mode = ""

while True:
    mode = str(input("Please type in 'normal' or 'survival' to select the cleaning mode: "))
    mode = mode.lower()
    if mode == "normal" or mode == "survival":
        break

if mode == "survival":
    path = "learn2clean/datasets/gbsg.csv"
    json_path = 'C:/Users/yosef/OneDrive/Desktop/Learn2Clean/python-package/config.json'
    dataset = pd.read_csv(path)
    time_column = "age"
    event_column = "status"
    model = ""
    while True:
        model = input("Please choose 'RSF', 'COX' or 'NN': ")
        model = model.upper()
        if model == 'RSF' or model == 'COX' or model == 'NN':
            break   

    l2c = survival_ql.SurvivalQlearner(dataset=dataset, time_col=time_column, event_col=event_column, goal=model, json_path=json_path, threshold=0.6)

    job = ""
    while True:
        job = str(input("Please choose 'L' for Learn2Clean, 'R' for Random, 'C' for Custom Pipeline Design or 'N' for a No Preparation job: ")).upper()
        if job == 'L' or job == 'R' or job == 'C' or job == 'N':
            break

    if job == "L":
        l2c.Learn2Clean()
    elif job == "R":
        repeat = eval(input("Please enter the number of random experiments: ")) # TODO Allow user to choose number of random trials
        l2c.random_cleaning(loop=repeat)
    elif job == 'C':
        pipelines_file_path = str(input("Please enter pipelines file name: "))
        pipelines = open(pipelines_file_path, 'r')
        l2c.custom_pipeline(pipelines, model)
    else:
        l2c.no_prep()

else:
    path = ["../datasets/titanic/titanic_train.csv","../datasets/titanic/test.csv"]
    hr = rd.Reader(sep=',',verbose=False, encoding=False) 
    dataset = hr.train_test_split(path, 'Survived')
    l2c_c1assification1 = ql.Qlearner(dataset = dataset,goal='CART', target_goal='Survived',threshold = 0.6, target_prepare=None, file_name = 'titanic_example', verbose = False)
    l2c_c1assification1.learn2clean()

'''
Survival_Qlearner Results against no Qlearner (Cox Model):
----------------------------------------------------------
    Dataset: subset.csv 
    
    Survival_Qlearner: c_index => 0.775885208424202

    Without Qlearning: c_index => 0.6479159251396854

  *********************************************************

    Dataset: gbsg.csv (Breast Cancer Dataset)


    ('None', 'random', 'COX', 'UC -> DBT -> MR -> COX', 'C-Index', 'Quality Metric: ', 0.8181818181818182)

    Survival_Qlearner: c_index => 0.912126155519304

    Without Qlearning: c_index => 0.7323545405111473
    

'''