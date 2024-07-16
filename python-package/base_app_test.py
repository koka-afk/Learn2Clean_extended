import learn2clean.loading.reader as rd 
import learn2clean.qlearning.qlearner as ql
# the results of learn2clean cleaning are stored in 'titanic_example'_results.txt in 'save' directory

titanic = ["../datasets/titanic/titanic_train.csv","../datasets/titanic/test.csv"]
hr=rd.Reader(sep=',',verbose=False, encoding=False) 
dataset=hr.train_test_split(titanic, 'Survived')


# Learn2clean finds the best strategy LOF -> CART for maximal accuracy : 0.7235772357723578 for CART
# in  234.35 seconds
# The best strategy is stored in EOF of 'titanic_example_results.txt' in 'save' directory as
# ('titanic_example', 'learn2clean', 'CART', 'Survived', None, 'LOF -> CART', 'accuracy', 0.7235772357723578, 234.34766507148743)

l2c_c1assification1=ql.Qlearner(dataset = dataset,goal='CART', target_goal='Survived',threshold = 0.6, target_prepare=None, file_name = 'titanic_example', verbose = False)
l2c_c1assification1.learn2clean()