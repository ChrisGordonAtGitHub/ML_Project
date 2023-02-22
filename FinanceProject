import tushare as ts

import numpy as np
import matplotlib.pyplot as plt

import warnings

from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectFromModel


warnings.filterwarnings('ignore')

dataFrame = ts.get_today_all()
originalColumns = dataFrame.columns
stocksData = dataFrame
ChineseLabels = ['代码','名称','涨跌幅','现价','开盘价','最高价','最低价','昨日收盘价','成交量','换手率','交易额','市盈率','市净率','总市值','流通市值']
stocksData.columns = ChineseLabels
stocks.to_excel('D:/Chris/Tool/12-Python&IDE/ML/Slf/RealTimeStockData20230222.xlsx')
X = stocks.loc[:,'现价':'流通市值'].values
#y = stocks.loc[:,'涨跌幅'].values
y = stocks['涨跌幅'].values
print(X.shape, y.shape)

scores = cross_val_score(MLPRegressor(random_state=8),X,y,cv=3)
print('模型平均分：{:.4f}'.format(scores.mean()))

pipe = make_pipeline(StandardScaler(), MLPRegressor(random_state=8, hidden_layer_sizes=(100,100))
scores = cross_val_score(pipe,X,y,cv=3)
print(scores)
print('模型平均分：{:.4f}'.format(scores.mean()))

pipe = make_pipeline(StandardScaler(), SelectFromModel(RandomForestRegressor(random_state=8)), MLPRegressor(random_state=8, hidden_layer_sizes=(100,100)))
pipe.steps
scores = cross_val_score(pipe,X,y,cv=3)
print('模型平均分：{:.4f}'.format(scores.mean()))

pipe.fit(X,y)
mask = pipe.named_steps['selectfrommodel'].get_support()
print(mask)

params = [{'reg':[MLPRegressor(random_state=8)],
           'scaler':[StandardScaler()],
           'reg__hidden_layer_sizes':[(10,),(50,),(100,),(100,100),(200,),(200,200),(600,),(800,800),(1000,1000)],
           'reg__solver':['sgd','lbfgs','adam'],
           'reg__activation':['identity','logistic','tanh','relu']},
          {'reg':[RandomForestRegressor(random_state=8,n_jobs=-1)],
           'scaler':[None],
           'reg__n_estimators':[10,50,100,200,300,500,600,800,1000]}]
pipe = Pipeline([('scaler',StandardScaler()),('reg',MLPRegressor())])
grid = GridSearchCV(pipe, params, cv=3, n_jobs=-1)
grid.fit(X,y)
print('最佳模型是：\n{}'.format(grid.best_params_))
print('\n模型最佳得分是：{:.6f}'.format(grid.best_score_))

params = [{'reg':[SVR(),SVC(random_state=8),LinearSVR(random_state=8),LinearSVC(random_state=8)],
           'scaler':[StandardScaler()],
           'reg__kernel':['linear','poly','rbf'],
           'reg__degree':[3,6],
           'reg__gamma':[1,0.1,10],
           'reg__C':[1,2]},
          {'reg':[RandomForestRegressor(random_state=8,n_jobs=-1)],
           'scaler':[None],
           'reg__n_estimators':[10,50,100,200]}]
pipe = Pipeline([('scaler',StandardScaler()),('reg',MLPRegressor())])
grid = GridSearchCV(pipe, params, cv=3, n_jobs=-1)
grid.fit(X,y)
print('最佳模型是：\n{}'.format(grid.best_params_))
print('\n模型最佳得分是：{:.6f}'.format(grid.best_score_))

# bins = np.linspace(-25, 36, 8)
# target_bin = np.digitize(y, bins=bins)
# print(target_bin)
