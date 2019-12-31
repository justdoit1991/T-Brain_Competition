# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:35:50 2019

@author: Felix
"""

# 刪除過去內容
#reset

# 指定資料讀取路徑
file_path='C:/Users/Felix/Google 雲端硬碟/鼎新相關/1. 工作 - 專案&客戶拜訪&例會/競賽/玉山銀行人工智慧公開挑戰賽/data/'

##############################  匯入所需的套件  ##############################

# 匯入資料整理與分析所需的模組
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler,binarize
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

##############################  讀取資料表  ##############################

# 顧客基本屬性資料
TBN_CIF=pd.read_csv(file_path+'TBN_CIF.csv',engine='python')
# 顧客網頁瀏覽行為
TBN_CUST_BEHAVIOR=pd.read_csv(file_path+'TBN_CUST_BEHAVIOR.csv',engine='python')
# 顧客最近一次交易時間
TBN_RECENT_DT=pd.read_csv(file_path+'TBN_RECENT_DT.csv',engine='python')
# 顧客信用卡核卡資料
TBN_CC_APPLY=pd.read_csv(file_path+'TBN_CC_APPLY.csv',engine='python')
# 顧客外匯交易資料	
TBN_FX_TXN=pd.read_csv(file_path+'TBN_FX_TXN.csv',engine='python')
# 顧客信貸申請資料
TBN_LN_APPLY=pd.read_csv(file_path+'TBN_LN_APPLY.csv',engine='python')
# 顧客信託類產品交易資料	
TBN_WM_TXN=pd.read_csv(file_path+'TBN_WM_TXN.csv',engine='python')
# 上傳結果資料檔	
TBN_Y_ZERO=pd.read_csv(file_path+'TBN_Y_ZERO.csv',engine='python')
print('read data','finished !') # print出讀取資料完畢的通知

##############################  探索資料  ##############################

'''
## 1.利用畫圖來看申辦信用卡次數的分佈 
CC_count=TBN_CC_APPLY.groupby('CUST_NO')['TXN_DT'].count() # 先計算每個人辦過信用卡的次數
CC_dist=pd.crosstab(CC_count,columns='TXN_DT')  # 再找出辦過信用卡次數的分配
plt.bar(CC_dist.index,height=CC_dist['TXN_DT'], width=1)  # 畫圖
plt.show()
## 由此可知大多數的人都只申辦一次信用卡

## 4.利用次數分配探討風險屬性跟平均交易次數之間的關係  
# 抓出顧客風險屬性
WM_mean=round(TBN_WM_TXN.groupby('CUST_NO')['CUST_RISK_CODE'].mean()).reset_index()
# 計算顧客交易次數
WM_count=TBN_WM_TXN.groupby('CUST_NO')['WM_TXN_AMT'].count().reset_index()
WM_count.rename(columns={'WM_TXN_AMT':'count'},inplace=True)
# 合併資料
WM_comb=WM_mean.merge(WM_count,on=['CUST_NO'],how='left')
# 重新定義顧客風險屬性1~3合併為同一組 , 4自己一組
WM_comb['CUST_RISK_CODE']=np.where(WM_comb['CUST_RISK_CODE']<=3,1,2)
# 計算不同顧客風險屬性的平均交易次數
WM_comb.groupby('CUST_RISK_CODE')['count'].mean()
## 由此可知風險屬性跟平均交易次數有關
'''

##############################  整理資料  ##############################
########################## 1.整理顧客基本資料 ##########################

# 重新定義子女數的LEVEL(大於3個皆定義為第4類別)
TBN_CIF['CHILDREN_NUM_LEVEL']=np.where(TBN_CIF['CHILDREN_CNT']>3,4,TBN_CIF['CHILDREN_CNT'])
# 新增開戶距離現在的時間(單位:年)
split_point=9537  # 定義一個時間分界點
TBN_CIF['CUST_START_NOW_DISTANCE']=TBN_CIF['CUST_START_DT'].apply(lambda x : (split_point-x)/365)
# 根據開戶距離現在的時間分不同LEVEL
TBN_CIF['CUST_START_NOW_LEVEL']=np.where(TBN_CIF['CUST_START_NOW_DISTANCE']>20,6,
                                np.where(TBN_CIF['CUST_START_NOW_DISTANCE']>15,5,
                                np.where(TBN_CIF['CUST_START_NOW_DISTANCE']>10,4,
                                np.where(TBN_CIF['CUST_START_NOW_DISTANCE']>5,3,
                                np.where(TBN_CIF['CUST_START_NOW_DISTANCE']>1,2,1)))))
# 將男女轉換成1＆0
mappings = {'F':0, 'M':1}
TBN_CIF.GENDER_CODE.replace(mappings, inplace=True)
# 最後排除不需要的欄位
TBN_CIF=TBN_CIF.drop(['CHILDREN_CNT','CUST_START_DT','WORK_MTHS'],axis=1)
print('TBN_CIF','finished !') # print出清洗完畢的通知

########################## 2.整理顧客過去交易行為 ##########################

start=9448  # 定義一個時間起始點=開始觀察時間　
# 新增過去是否進行各交易行為的指標與距離開始觀察的時間　
SUB_TBN_RECENT_DT=TBN_RECENT_DT.drop('CUST_NO',axis=1)
RECENT_APLLY=pd.DataFrame(np.where(SUB_TBN_RECENT_DT>0,1,0)).rename(columns={0:'RECENT_CC_APLLY',1:'RECENT_FX_APLLY',2:'RECENT_LN_APLLY',3:'RECENT_WM_APLLY'},inplace=False)
RECENT_DISTANCE=pd.DataFrame(np.where(SUB_TBN_RECENT_DT>0,round((start-SUB_TBN_RECENT_DT)/365,2),SUB_TBN_RECENT_DT)).rename(columns={0:'RECENT_CC_DISTANCE',1:'RECENT_FX_DISTANCE',2:'RECENT_LN_DISTANCE',3:'RECENT_WM_DISTANCE'},inplace=False)
# 合併過去交易紀錄
RECENT_RECORD=pd.concat([TBN_RECENT_DT['CUST_NO'],RECENT_APLLY,RECENT_DISTANCE],axis=1,join_axes=[TBN_RECENT_DT.index])

# 刪除接下來不需要的table節省記憶體空間
del SUB_TBN_RECENT_DT,RECENT_APLLY,RECENT_DISTANCE

print('RECENT_RECORD','finished !') # print出清洗完畢的通知

########################## 3.整理顧客信託類產品交易資料 ##########################

# 創立時間分界點的字典
split_point_dict={1:9537 ,2:9568}
for key in split_point_dict.keys():  # 利用迴圈整理不同時間分界點的交易行為
    
    # 先抓出不重覆的CUST_NO , 並重新定義欄位名稱　　
    unique_CUST_NO=pd.DataFrame(TBN_WM_TXN['CUST_NO'].unique())
    unique_CUST_NO.rename(columns={0:'CUST_NO'},inplace=True)
    
    # 根據時間分界點切割資料(抓出前90天)
    data1=TBN_WM_TXN[TBN_WM_TXN['TXN_DT'].between(split_point_dict[key]-89,split_point_dict[key])]
    # 再將前90天的資料切成3份
    data1_1=data1[data1['TXN_DT'].between(split_point_dict[key]-29,split_point_dict[key])]
    data1_2=data1[data1['TXN_DT'].between(split_point_dict[key]-59,split_point_dict[key]-30)]
    data1_3=data1[data1['TXN_DT'].between(split_point_dict[key]-89,split_point_dict[key]-60)]

    # 抓出前90天顧客的風險屬性和交易的信託性質
    CUST_RISK_CODE=round(data1.groupby('CUST_NO')['CUST_RISK_CODE'].mean()).reset_index()
    INVEST_TYPE_CODE=round(data1.groupby('CUST_NO')['INVEST_TYPE_CODE'].mean()).reset_index()

    #計算前90天內顧客最早交易日期、最近交易日期、總交易次數
    LAST_90DAYS_WM_RECORD =data1.groupby('CUST_NO',as_index=False)['TXN_DT'].agg({'min':'min','max':'max','LAST_90DAYS_WM_NUM':'count'})
    # 新增前90天是否有進行交易的指標
    LAST_90DAYS_WM_RECORD['LAST_90DAYS_WM_APLLY']=1
    # 計算最近交易日期與現在時間的間隔、前90天每次交易之間的平均間隔
    LAST_90DAYS_WM_RECORD['split_point']=split_point_dict[key]
    LAST_90DAYS_WM_RECORD['LATEST_NOW_INTERVAL']=LAST_90DAYS_WM_RECORD['split_point']-LAST_90DAYS_WM_RECORD['max']
    LAST_90DAYS_WM_RECORD['max-min']=LAST_90DAYS_WM_RECORD['max']-LAST_90DAYS_WM_RECORD['min']
    LAST_90DAYS_WM_RECORD['LAST_90DAYS_AVG_INTERVAL']=LAST_90DAYS_WM_RECORD['max-min']/LAST_90DAYS_WM_RECORD['LAST_90DAYS_WM_NUM']
    # 最後排除不需要的欄位
    LAST_90DAYS_WM_RECORD=LAST_90DAYS_WM_RECORD.drop(['min','max','split_point','max-min'],axis=1)

    #　計算每個區間資料中每位顧客的交易次數
    LAST_30_WM_NUM=data1_1.groupby('CUST_NO',as_index=False)['WM_TXN_AMT'].agg({'LAST_30_WM_NUM':'count'})
    LAST_60_30_WM_NUM=data1_2.groupby('CUST_NO',as_index=False)['WM_TXN_AMT'].agg({'LAST_60_30_WM_NUM':'count'})
    LAST_90_60_WM_NUM=data1_3.groupby('CUST_NO',as_index=False)['WM_TXN_AMT'].agg({'LAST_90_60_WM_NUM':'count'})
    
    # 根據時間分界點切割資料(抓出後30天)
    if split_point_dict[key]<9538:
        data2=TBN_WM_TXN[TBN_WM_TXN['TXN_DT'].between(split_point_dict[key]+1,split_point_dict[key]+30)]
        # 找出後30天有交易的顧客+更改欄位名稱,並新增後30天是否有進行交易的指標
        RESULT=pd.DataFrame(data2['CUST_NO'].unique()).rename(columns={0:'CUST_NO'},inplace=False)
        RESULT['APPLY_WM']=1
        # 合併所有特徵
        WM_RECORD_FEATURE=unique_CUST_NO.merge(CUST_RISK_CODE,on=['CUST_NO'],how='left').merge(INVEST_TYPE_CODE,on=['CUST_NO'],how='left').merge(LAST_90DAYS_WM_RECORD,on=['CUST_NO'],how='left').merge(LAST_30_WM_NUM,on=['CUST_NO'],how='left').merge(LAST_60_30_WM_NUM,on=['CUST_NO'],how='left').merge(LAST_90_60_WM_NUM,on=['CUST_NO'],how='left').merge(RESULT,on=['CUST_NO'],how='left')
    else:
        WM_RECORD_FEATURE2=unique_CUST_NO.merge(CUST_RISK_CODE,on=['CUST_NO'],how='left').merge(INVEST_TYPE_CODE,on=['CUST_NO'],how='left').merge(LAST_90DAYS_WM_RECORD,on=['CUST_NO'],how='left').merge(LAST_30_WM_NUM,on=['CUST_NO'],how='left').merge(LAST_60_30_WM_NUM,on=['CUST_NO'],how='left').merge(LAST_90_60_WM_NUM,on=['CUST_NO'],how='left')
 
print('WM_RECORD_FEATURE','finished !') # print出清洗完畢的通知

##############################  合併分析資料＆拆解訓練、驗證、測試資料  ##############################

for i in range(1,3): # 利用迴圈合併不同時間分界點的完整資料
    
    # 合併WM所有相關資料
    if i==1:
        WM_comb=pd.DataFrame(TBN_Y_ZERO['CUST_NO']).merge(TBN_CIF,on=['CUST_NO'],how='outer').merge(RECENT_RECORD,on=['CUST_NO'],how='outer').merge(WM_RECORD_FEATURE,on=['CUST_NO'],how='outer')
    else:
        WM_comb=pd.DataFrame(TBN_Y_ZERO['CUST_NO']).merge(TBN_CIF,on=['CUST_NO'],how='left').merge(RECENT_RECORD,on=['CUST_NO'],how='left').merge(WM_RECORD_FEATURE2,on=['CUST_NO'],how='left')
        
    # 新增過去或前90天是否有交易行為
    WM_comb['ever_WM_apply']=np.where((WM_comb['RECENT_WM_APLLY']==1)|(WM_comb['LAST_90DAYS_WM_APLLY']==1),1,0)
    # 補遺漏值
    WM_comb.RECENT_CC_APLLY.fillna(0,inplace=True)
    WM_comb.RECENT_FX_APLLY.fillna(0,inplace=True)
    WM_comb.RECENT_LN_APLLY.fillna(0,inplace=True)
    WM_comb.RECENT_WM_APLLY.fillna(0,inplace=True)

    WM_comb.RECENT_CC_DISTANCE.fillna(99,inplace=True)
    WM_comb.RECENT_FX_DISTANCE.fillna(99,inplace=True)
    WM_comb.RECENT_LN_DISTANCE.fillna(99,inplace=True)
    WM_comb.RECENT_WM_DISTANCE.fillna(99,inplace=True)

    WM_comb.LAST_90DAYS_WM_NUM.fillna(0,inplace=True)
    WM_comb.LAST_90DAYS_WM_APLLY.fillna(0,inplace=True)

    WM_comb.LATEST_NOW_INTERVAL.fillna(99,inplace=True)
    WM_comb.LAST_90DAYS_AVG_INTERVAL.fillna(99,inplace=True)

    WM_comb.LAST_30_WM_NUM.fillna(0,inplace=True)
    WM_comb.LAST_60_30_WM_NUM.fillna(0,inplace=True)
    WM_comb.LAST_90_60_WM_NUM.fillna(0,inplace=True)   
    WM_comb.ever_WM_apply.fillna(0,inplace=True)
    
    if i==1:
        WM_comb.APPLY_WM.fillna(0,inplace=True)
        WM_data=WM_comb
    else:
        WM_data2=WM_comb

print('WM_data','finished !') # print出清洗完畢的通知


## 配對資料
# 先將遺漏值全部補0
comp_data=WM_data.fillna(0,inplace=False)
# 區分為目標資料跟即將配對資料
treated_df=comp_data[comp_data['CUST_NO'].isin(TBN_Y_ZERO['CUST_NO'])].drop(['CUST_NO','CUST_START_NOW_DISTANCE','APPLY_WM'],axis=1)
non_treated_df=comp_data[~comp_data['CUST_NO'].isin(TBN_Y_ZERO['CUST_NO'])].drop(['CUST_NO','CUST_START_NOW_DISTANCE','APPLY_WM'],axis=1)

# 定義配對資料的函數
def get_matching_pairs(treated_df, non_treated_df, scaler=True):

    treated_x = treated_df.values
    non_treated_x = non_treated_df.values
    if scaler == True:
        scaler = StandardScaler()
    if scaler:
        scaler.fit(treated_x)
        treated_x = scaler.transform(treated_x)
        non_treated_x = scaler.transform(non_treated_x)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(non_treated_x)
    DISTANCEs, indices = nbrs.kneighbors(treated_x)
    indices = indices.reshape(indices.shape[0])
    matched = non_treated_df.ix[indices]
    return matched

matched_df = get_matching_pairs(treated_df, non_treated_df)

print('matched_data','finished !') # print出配對資料完畢的通知


# 抓出最後的訓練集和測試集,並丟棄CUST_NO
data3=WM_data[WM_data.index.isin(matched_df.index)].drop('CUST_NO', axis = 1)
data4=WM_data[WM_data.index.isin(treated_df.index)].drop('CUST_NO', axis = 1)
train_data=data3
#test_data=data4
test_data=WM_data2.drop('CUST_NO', axis = 1)

# 切割出Ｘ＆ｙ的訓練和測試集
X_train = train_data.drop('APPLY_WM', axis = 1)
y_train = train_data[['APPLY_WM']].values.reshape(-1,)
#X_test = test_data.drop('APPLY_WM', axis = 1)
#y_test = test_data[['APPLY_WM']].values.reshape(-1,)
X_test = test_data

print('train_valid_test_data','finished !') # print出切割資料完畢的通知

##############################  建立模型  ##############################
'''
# 設定cv個數
cv_num=5

#設定XGBoost的參數搜索範圍(只搜索指定的参数)
xgb_param = {
 'learning_rate': [0.05, 0.1],               # 學習率，也就是梯度下降法中的步長。太小的话，训练速度太慢，而且容易陷入局部最優点。通常是0.0001到0.1之间 
 'n_estimators' : [30, 50, 100, 150],        # 樹的個數。并非越多越好，通常是50到1000之间。
 'max_depth' : [2, 3],                       # 每棵樹的最大深度。太小会欠擬合，太大过擬合。正常值是3到10。
 'min_child_weight' : [3, 4, 5, 6, 7],       # 決定最小葉子節點樣本權重和。當它的值較大時，可以避免模型學習到局部的特殊樣本。但如果這個值過高，會導致欠擬合。
 'subsample' : [0.7, 0.8, 0.9],              # 随機抽樣的比例
 'colsample_bytree' : [0.4, 0.5, 0.6, 0.8],  # 訓練每個樹時用的特徵的數量。1表示使用全部特徵，0.5表示使用一半的特徵
 'gamma' : [0],                              # 在節點分裂時，只有在分裂後損失函數的值下降了，才會分裂這個節點。
 'reg_alpha' : [1],                          # L1 正則化項的權重係數，越大模型越保守，用来防止过拟合。一般是0到1之间。(和Lasso regression類似)。
 'reg_lambda' : [3, 4],                      # L2 正則化項的權重係數，越大模型越保守，用来防止过拟合。一般是0到1之间。(和Ridge regression類似)。
 'objective' : ['binary:logistic'],          # 定義學習任務及相應的學習目標
 'nthread' : [-1],                           # cpu 線程数
 'scale_pos_weight' : [2,3],                 # 各類樣本十分不平衡時，把這個參數設置為一個正數，可以使算法更快收斂。典型值是sum(negative cases) / sum(positive cases)
 'seed' : [8765]}                            # 隨機種子

## 建立XGBoost模型
#分类器使用 XGBoost
xgbc = xgb.XGBClassifier()

# 使用RandomizedSearch的交叉验证来选择参数
randomized = RandomizedSearchCV(xgbc,xgb_param,iid=True,cv=cv_num,scoring='f1',n_jobs=-1,n_iter=30)
randomized.fit(X_train, y_train)
# 使用GridSearch的交叉验证来选择参数
#grid = GridSearchCV(xgbc,xgb_param,cv=cv_num,scoring='f1',n_jobs=-1)
#grid.fit(X_train, y_train)

bxgbc = randomized.best_estimator_ # 再利用最佳參數訓練模型
#bxgbc = grid.best_estimator_  # 再利用最佳參數訓練模型
'''
# 目前最佳參數
bxgbc = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=3, min_child_weight=3, missing=None, n_estimators=50,
       n_jobs=1, nthread=-1, objective='binary:logistic', random_state=0,
       reg_alpha=1, reg_lambda=4, scale_pos_weight=3, seed=8765,
       silent=True, subsample=0.7)

bxgbc.fit(X_train,y_train) #训练模型
y_pred = bxgbc.predict(X_test) #预测模型
y_pred_proba = bxgbc.predict_proba(X_test) #返回機率
xgb.plot_importance(bxgbc)  # 畫出特徵的重要性
plt.show() # 图形显示

# 根據新定義的閥值分類 => 大於0.5為會購買信託 , 小於則為不會購買信託
y_new_pred=binarize(y_pred_proba , threshold=0.5)[:,1]
#參考網址:https://reurl.cc/jkoGp

print('model','finished !') # print出模型建立完畢的通知

#######################  儲存預測結果  #######################

# 將結果填入上傳檔 , 並輸出csv檔(且不show出index)
TBN_Y_ZERO['WM_IND']=y_new_pred
TBN_Y_ZERO.to_csv('C:/Users/Felix/Desktop/'+'TBN_Y_ZERO.csv',index=False)

print('result','finished !') # print出結果儲存完畢的通知

####################  訓練集和測試集在模型的表現  ####################

print('模型預測會有',y_new_pred.sum(),'人會購買信託') 

# 將訓練集丟入模型預測
X_test = X_train
y_test = y_train

bxgbc.fit(X_train,y_train) #训练模型
y_pred = bxgbc.predict(X_test) #预测模型
y_pred_proba = bxgbc.predict_proba(X_test) #返回機率

# 根據新定義的閥值分類 => 大於0.5為會購買信託 , 小於則為不會購買信託
y_new_pred=binarize(y_pred_proba , threshold=0.5)[:,1]

print('################# 訓練集的預測結果 #################') 

# 建立混淆矩陣
from sklearn.metrics import confusion_matrix
C=confusion_matrix(y_test,y_new_pred)
print(C)

# 計算f1_score
print(classification_report(y_test,y_new_pred))

#計算Precision
print('Precision=',C[1,1]/C[:,1].sum())

#計算Recall
print('Recall=',C[1,1]/C[1,].sum())


# 用預測的30,000人當測試集
test_data=data4
# 切割出Ｘ＆ｙ的測試集
X_test = test_data.drop('APPLY_WM', axis = 1)
y_test = test_data[['APPLY_WM']].values.reshape(-1,)

bxgbc.fit(X_train,y_train) #训练模型
y_pred = bxgbc.predict(X_test) #预测模型
y_pred_proba = bxgbc.predict_proba(X_test) #返回機率

# 根據新定義的閥值分類 => 大於0.5為會購買信託 , 小於則為不會購買信託
y_new_pred=binarize(y_pred_proba , threshold=0.5)[:,1]

print('################# 測試集的預測結果 #################') 

# 建立混淆矩陣
from sklearn.metrics import confusion_matrix
C=confusion_matrix(y_test,y_new_pred)
print(C)

# 計算f1_score
print(classification_report(y_test,y_new_pred))

#計算Precision
print('Precision=',C[1,1]/C[:,1].sum())

#計算Recall
print('Recall=',C[1,1]/C[1,].sum())