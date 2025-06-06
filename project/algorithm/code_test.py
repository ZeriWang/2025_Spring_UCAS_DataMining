import pandas as pd
import numpy as np
import category_encoders as ce
import os
import gc
from tqdm import *
# 核心模型使用第三方库
import lightgbm as lgb
# 新增TabNet模型依赖
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
# 交叉验证所使用的第三方库
from sklearn.model_selection import StratifiedKFold, KFold
# 评估指标所使用的的第三方库
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
import copy
import datetime
from chinese_calendar import is_workday
from numpy import nan
# 忽略报警所使用的第三方库
import pandas._testing as tm
import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_row', 1000)

# 读取训练集和测试集
train_data = pd.read_csv("/home/zeriwang/2025Spring/DataMining/2025_Spring_UCAS_DataMining/project/data/train.csv")
test_data = pd.read_csv("/home/zeriwang/2025Spring/DataMining/2025_Spring_UCAS_DataMining/project/data/evaluation_public.csv")

# 添加标识并合并数据
train_data['istest'] = 0
test_data['istest'] = 1
data = pd.concat([train_data, test_data]).reset_index(drop=True)

data['id_by_me'] = pd.Series(range(len(data)))

# 删除相同的特征
del data['ip_type']

# 特征工程

#风险系数高的情况
# 标记高风险地点
data['is_forei/unknown'] = data['op_city'].apply(lambda x: 1 if x in['国外','未知'] else 0)
# 标记HTTP状态码异常
data['is_fail_code']=data['http_status_code'].apply(lambda x:0 if x==200 else 1)
data['is_code_5']=data['http_status_code'].apply(lambda x:1 if x in [500,502] else 0)
# 标记登录URL
data['is_login_url'] = data['url'].apply(lambda x:1 if x in ['xxx.com/getVerifyCode','xxx.com/getLoginType'] else 0)


# 转换时间格式
data['op_datetime'] = pd.to_datetime(data['op_datetime'])
# 提取日期部分
data['day']=data['op_datetime'].astype(str).apply(lambda x:str(x)[5:10])
# 提取小时
data['hour'] = data['op_datetime'].dt.hour
# 提取星期几(1-7)
data['weekday'] = data['op_datetime'].dt.weekday+1

data = data.sort_values(by=['user_name', 'op_datetime']).reset_index(drop=True)

# 时间周期性特征（三角函数转换）
data['hour_sin'] = np.sin(data['hour']/24*2*np.pi)
data['hour_cos'] = np.cos(data['hour']/24*2*np.pi)

data['op_day'] = data['op_datetime'].astype(str).apply(lambda x:str(x)[8:10])

data['min'] = data['op_datetime'].apply(lambda x: int(str(x)[-5:-3]))
data['min_sin'] = np.sin(data['min']/60*2*np.pi)
data['min_cos'] = np.cos(data['min']/60*2*np.pi)

# 计算用户两次操作之间的时间差
data['diff_last_1'] = data.groupby('user_name')['op_datetime'].transform(lambda i:i.diff(1)).dt.total_seconds()/60
data['diff_last_2'] = data.groupby('user_name')['op_datetime'].transform(lambda i:i.diff(2)).dt.total_seconds()/60

train_data = data[data['istest']==0]
test = data[data['istest']==1]

# 计算用户下一次操作的时间差（仅在训练集）
train_data['diff_next'] = -(train_data.groupby('user_name')['op_datetime'].transform(lambda i:i.diff(-1))).dt.total_seconds()/60
data=pd.merge(data,train_data[['diff_next','id_by_me']],how='left', on='id_by_me')

# 对多个特征计算时间差相关的统计量
fea = ['user_name', 'department', 'ip_transform', 'device_num_transform', 'browser_version', 'browser',
          'os_type', 'os_version',  'op_city', 'log_system_transform', 'url']

for col in fea:
    # 计算每个特征分组下时间差的统计量
    data[col+'_diff1_mean'] = data.groupby(col)['diff_last_1'].transform('mean')
    data[col+'_diff1_std'] = data.groupby(col)['diff_last_1'].transform('std')
    data[col+'_diff1_max'] = data.groupby(col)['diff_last_1'].transform('max')
    data[col+'_diff1_min'] = data.groupby(col)['diff_last_1'].transform('min')

for col in fea:
    data[col+'_diff_next_mean'] = data.groupby(col)['diff_next'].transform('mean')
    data[col+'_diff_next_std'] = data.groupby(col)['diff_next'].transform('std')
    data[col+'_diff_next_max'] = data.groupby(col)['diff_next'].transform('max')
    data[col+'_diff_next_min'] = data.groupby(col)['diff_next'].transform('min')

#data.drop(['browser_version_diff1_min','browser_diff1_min','os_type_diff1_min','os_version_diff1_min'],axis=1,inplace=True)

data=data.fillna(-999)


def is_fail_code(x):
    if x==200:
        return 0
    else:
        return 1

#data['is_fail_code']=data['http_status_code'].apply(is_fail_code)

data['is_fail_usr']=data['user_name'].apply(lambda x:1 if x == "-999" else 0)#登陆失败

# 周末标记
def isweekend(x):
    if(x<6):
        return 0
    else:
        return 1

data['isweekend']=data['weekday'].apply(isweekend)

# 夜间标记(晚8点到早7点)
def isnight(x):
    if (x>7)and(x<20):
        return 0
    else:
        return 1

data['isnight'] = data['hour'].apply(isnight)

#节假日放假
holiday = ['01-31','02-01', '02-02', '02-03', '02-04', '02-05', '02-06',
           '04-03','04-04', '04-05', '05-01', '05-02', '05-03', '05-04',
           '06-03', '06-04', '06-05']
def if_holiday(x):
    if x in holiday:
        return 1
    else:
        return 0
data['isholiday'] = data['op_datetime'].apply(lambda x:if_holiday(str(x)[5:10]))

#调休
adjust = ['01-29', '01-30','04-02', '04-24','05-07']
def if_adjust(x):
    if x in adjust:
        return 1
    else:
        return 0
data['is_adjust'] = data['day'].apply(if_adjust)

# 非工作日标记(周末且非调休，或节假日)
data['is_not_work'] = data['isweekend'].astype(bool)&(~data['is_adjust'])|(data['isholiday'].astype(bool))



time_fea=['hour','weekday','min','isnight','isholiday','is_not_work']

for col in time_fea:
    data[col+'_diff1_mean_u'] = data.groupby(['user_name',col])['diff_last_1'].transform('mean')
    data[col+'_diff1_std_u'] = data.groupby(['user_name',col])['diff_last_1'].transform('std')

for col in time_fea:
    data[col+'_diff1_next_mean_u'] = data.groupby(['user_name',col])['diff_next'].transform('mean')
    data[col+'_diff1_next_std_u'] = data.groupby(['user_name',col])['diff_next'].transform('std')

del data['diff_next']




cols = ['id_by_me','user_name','ip_transform', 'device_num_transform',
       'browser_version', 'browser', 'os_type', 'os_version','http_status_code','op_city',
        'log_system_transform','url','op_datetime']

tmp=data[cols]

tmp['op_day'] = tmp['op_datetime'].dt.date

tmp = tmp.groupby(['user_name','op_day'],as_index=False).agg({'id_by_me':list,'ip_transform':list, 'device_num_transform':list,
       'browser_version':list, 'browser':list, 'os_type':list, 'os_version':list,'http_status_code':list,'op_city':list,
        'log_system_transform':list,'url':list})

def get_which_time(col_unique,fea):
    fea_dict = dict.fromkeys(col_unique,0)
    count_list=[]
    for i in range(len(fea)):
        fea_dict[fea[i]] = fea_dict[fea[i]]+1
        count_list.append(fea_dict[fea[i]])
    return count_list

for col in tqdm(['ip_transform', 'device_num_transform',
       'browser_version', 'browser', 'os_type', 'os_version','http_status_code','op_city',
        'log_system_transform','url']):
    col_unique=data[col].unique()
    tmp[col+'_countls'] = tmp[col].apply(lambda x:get_which_time(col_unique,x))

tmp=tmp.explode(['id_by_me', 'ip_transform',
       'device_num_transform', 'browser_version', 'browser', 'os_type',
       'os_version', 'http_status_code', 'op_city', 'log_system_transform',
       'url', 'ip_transform_countls',
       'device_num_transform_countls', 'browser_version_countls',
       'browser_countls', 'os_type_countls', 'os_version_countls',
       'http_status_code_countls', 'op_city_countls',
       'log_system_transform_countls', 'url_countls'])

tmp = tmp.reset_index(drop=True)

cols=['id_by_me','ip_transform_countls', 'device_num_transform_countls',
       'browser_version_countls', 'browser_countls', 'os_type_countls',
       'os_version_countls', 'http_status_code_countls', 'op_city_countls',
       'log_system_transform_countls', 'url_countls']

data=pd.merge(data,tmp[cols],on='id_by_me',how='left')

for col in ['ip_transform_countls', 'device_num_transform_countls',
       'browser_version_countls', 'browser_countls', 'os_type_countls',
       'os_version_countls', 'http_status_code_countls', 'op_city_countls',
       'log_system_transform_countls', 'url_countls']:
    data[col] = data[col].astype(int)






cols = ['id_by_me','user_name','ip_transform', 'device_num_transform','browser_version', 'browser', 'os_type', 
                 'os_version','http_status_code','op_city','log_system_transform','url','op_datetime']
tmp=data[cols]

#账号最近几次登陆
for x in range(1,30):
    tmp['usr_diff_last_'+str(x)] = tmp.groupby(['user_name'])['op_datetime'].transform(lambda i:i.diff(x)).dt.total_seconds()/60
merge_cols = [col for col in tmp.columns if '_diff_last_' in col]
tmp['ip_diff_list_30']=tmp[merge_cols].values.tolist()
tmp.drop(merge_cols,axis=1,inplace=True)

#账号最近几次登陆对应ip
for x in range(1,30):
    tmp['usr_last_ip'+str(x)] = tmp.groupby(['user_name'])['ip_transform'].transform(lambda i:i.shift(x))
merge_cols = [col for col in tmp.columns if '_last_' in col]
tmp['usr_ip_list_30']=tmp[merge_cols].values.tolist()
tmp.drop(merge_cols,axis=1,inplace=True)

def get_nunique_minute(diff_list,uni_list,minute):
    ls=[]
    for i in range(len(diff_list)):
        if diff_list[i]<minute:
            ls.append(uni_list[i])
        else:
            break
    return pd.Series(ls).nunique()

# 计算用户在不同时间窗口内使用的不同IP数量
tmp['ip_time_nui_6'] = tmp.apply(lambda row:get_nunique_minute(row['ip_diff_list_30'],row['usr_ip_list_30'],60*6),axis=1)

tmp['ip_time_nui_12'] = tmp.apply(lambda row:get_nunique_minute(row['ip_diff_list_30'],row['usr_ip_list_30'],60*12),axis=1)

tmp['ip_time_nui_24'] = tmp.apply(lambda row:get_nunique_minute(row['ip_diff_list_30'],row['usr_ip_list_30'],60*24),axis=1)

cols=[col for col in tmp.columns if 'ip_time_nui_'in col]

cols.append('id_by_me')

data=pd.merge(data,tmp[cols],on='id_by_me',how='left')




cross_cols=[]

#department_city
data['department_op_city'] = data['department'].astype(str)+data['op_city'].astype(str)
cross_cols.append('department_op_city')

#department_log_system_transform
data['department_log_system_transform'] = data['department'].astype(str)+data['log_system_transform'].astype(str)
#data['department_log_system_transform'] = label_encoder(data['department_log_system_transform'])
cross_cols.append('department_log_system_transform')

#browser_version_op_city
data['browser_version_op_city'] = data['browser_version'].astype(str)+data['op_city'].astype(str)
#data['browser_version_op_city'] = label_encoder(data['browser_version_op_city'])
cross_cols.append('browser_version_op_city')

#browser_op_city
data['browser_op_city'] = data['browser'].astype(str)+data['op_city'].astype(str)
#data['browser_op_city'] = label_encoder(data['browser_op_city'])
cross_cols.append('browser_op_city')

#browser_log_system_transform
data['browser_log_system_transform'] = data['browser'].astype(str)+data['log_system_transform'].astype(str)
#data['browser_log_system_transform'] = label_encoder(data['browser_log_system_transform'])
cross_cols.append('browser_log_system_transform')

#os_type_op_city
data['os_type_op_city'] = data['os_type'].astype(str)+data['op_city'].astype(str)
#data['os_type_op_city'] = label_encoder(data['os_type_op_city'])
cross_cols.append('os_type_op_city')

#os_type_log_system_transform
data['os_type_log_system_transform'] = data['os_type'].astype(str)+data['log_system_transform'].astype(str)
#data['os_type_log_system_transform'] = label_encoder(data['os_type_log_system_transform'])
cross_cols.append('os_type_log_system_transform')

#os_version_op_city
data['os_version_op_city'] = data['os_version'].astype(str)+data['op_city'].astype(str)
#data['os_version_op_city'] = label_encoder(data['os_version_op_city'])
cross_cols.append('os_version_op_city')

#os_type_log_system_transform
data['os_type_log_system_transform'] = data['os_type'].astype(str)+data['log_system_transform'].astype(str)
#data['os_type_log_system_transform'] = label_encoder(data['os_type_log_system_transform'])
cross_cols.append('os_type_log_system_transform')

#op_city_log_system_transform
data['op_city_log_system_transform'] = data['op_city'].astype(str)+data['log_system_transform'].astype(str)
#data['op_city_log_system_transform'] = label_encoder(data['op_city_log_system_transform'])
cross_cols.append('op_city_log_system_transform')


#departmen_url
data['op_city_log_system_transform'] = data['department'].astype(str)+data['log_system_transform'].astype(str)
#data['op_city_log_system_transform'] = label_encoder(data['op_city_log_system_transform'])
cross_cols.append('op_city_log_system_transform')




cols = ['ip_transform', 'device_num_transform',
       'browser_version', 'browser', 'os_type', 'os_version',
       'http_status_code', 'op_city', 'log_system_transform', 'url']

for col in cols:
    tmp = data[data['istest']==0].groupby(['user_name',col,'hour'])['is_risk'].count().reset_index()
    tmp.columns=['user_name',col,'hour',col+'_hour_count']
    data=pd.merge(data,tmp,on=['user_name',col,'hour'],how='left')


tmp = data[data['istest']==0].groupby(['user_name','is_not_work','hour'],as_index=False)['is_risk'].agg({'work_hour_count':'count'})
data = pd.merge(data,tmp,how='left',on=['user_name','is_not_work','hour'])

date_fea = ['weekday','isholiday']

for col in date_fea:
    tmp = data[data['istest']==0].groupby(['user_name',col,'hour'],as_index=False)['is_risk'].agg({col+'_count':'count'})
    data = pd.merge(data,tmp,how='left',on=['user_name',col,'hour'])





numeric_features = data.select_dtypes(include=[np.number])
categorical_features = data.select_dtypes(include=[object])

# 对分类特征进行序数编码
data[categorical_features.columns] = ce.OrdinalEncoder().fit_transform(data[categorical_features.columns])






# 分离训练和测试数据
train_data = data[data['istest']==0]
test = data[data['istest']==1]

test = test.sort_values('id').reset_index(drop=True)

# 按时间分割训练集和验证集
train = train_data[train_data['op_datetime']<'2022-04-01'].reset_index(drop=True)
val = train_data[train_data['op_datetime']>='2022-04-01'].reset_index(drop=True)

fea = [col for col in data.columns if col not in['id','index','id_by_me','op_datetime', 'op_day','day', 'op_month','is_risk', 'ts', 'ts1', 'ts2', 'diff_next']]

x_train = train[fea]
y_train = train['is_risk']

x_val = val[fea]
y_val = val['is_risk']

x_test = test[fea]
y_test = test['is_risk']

# LightGBM模型部分
importance = 0
pred_y = pd.DataFrame()
var_pre = pd.DataFrame()

lgb_score_list = []

seeds=2025

# LightGBM参数设置
params_lgb  = {
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 64,
    'verbose': -1,
    'seed': 2025,
    'n_jobs': -1,

    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 4,
    # 'min_child_weight': 10,
    "min_data_in_leaf":20
}

# TabNet模型预测结果保存
tabnet_pred_y = pd.DataFrame()
tabnet_var_pre = pd.DataFrame()
tabnet_score_list = []

# 预先初始化DataFrame索引，确保长度一致
tabnet_pred_y = pd.DataFrame(index=range(len(x_test)))
tabnet_var_pre = pd.DataFrame(index=range(len(x_train)))  # 修改为x_train的长度

kf = KFold(n_splits=5, shuffle=True, random_state=2025)
for i, (train_idx, val_idx) in enumerate(kf.split(x_train, y_train)):
    print('************************************ {} {}************************************'.format(str(i+1), str(seeds)))
    # 准备训练数据
    trn_x, trn_y, val_x, val_y = x_train.iloc[train_idx],y_train.iloc[train_idx], x_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # ====== LightGBM模型训练 ======
    train_data = lgb.Dataset(trn_x, trn_y)
    val_data = lgb.Dataset(val_x, val_y)
    
    # 训练模型
    model = lgb.train(params_lgb, train_data, valid_sets=[val_data], num_boost_round=20000,
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(2000)])

    # 预测并记录结果
    pred_y['fold_%d_seed_%d' % (i, seeds)] = model.predict(x_test)
    var_pre['fold_%d_seed_%d' % (i, seeds)] = model.predict(x_val)
    
    importance += model.feature_importance(importance_type='gain') / 5
    lgb_score_list.append(auc(val_y, model.predict(val_x)))
    
    # ====== TabNet模型训练 ======
    print("Training TabNet model for fold {}".format(i+1))
    
    # 保存原始索引以确保预测结果能正确对应
    original_val_idx = val_x.index
    
    # 转换为NumPy数组前确保没有NaN值
    trn_x_tabnet = trn_x.fillna(-999).copy()
    val_x_tabnet = val_x.fillna(-999).copy()
    x_test_tabnet = x_test.fillna(-999).copy()
    
    # 检查是否还有NaN值
    if trn_x_tabnet.isna().any().any():
        print("警告：训练数据中仍存在NaN值，将进一步处理")
        trn_x_tabnet = trn_x_tabnet.fillna(0)
    
    if val_x_tabnet.isna().any().any():
        print("警告：验证数据中仍存在NaN值，将进一步处理")
        val_x_tabnet = val_x_tabnet.fillna(0)
    
    if x_test_tabnet.isna().any().any():
        print("警告：测试数据中仍存在NaN值，将进一步处理")
        x_test_tabnet = x_test_tabnet.fillna(0)
    
    # 检查并移除非数值型列（object类型）
    object_cols = trn_x_tabnet.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"警告：发现对象类型列 {object_cols}，这些将从TabNet输入中移除")
        trn_x_tabnet = trn_x_tabnet.select_dtypes(exclude=['object'])
        val_x_tabnet = val_x_tabnet.select_dtypes(exclude=['object'])
        x_test_tabnet = x_test_tabnet.select_dtypes(exclude=['object'])
    
    # 强制转换所有列为数值型
    for col in trn_x_tabnet.columns:
        trn_x_tabnet[col] = pd.to_numeric(trn_x_tabnet[col], errors='coerce').fillna(0)
        val_x_tabnet[col] = pd.to_numeric(val_x_tabnet[col], errors='coerce').fillna(0)
        x_test_tabnet[col] = pd.to_numeric(x_test_tabnet[col], errors='coerce').fillna(0)
    
    # 确保数据类型为float64
    trn_x_tabnet = trn_x_tabnet.astype(np.float64)
    val_x_tabnet = val_x_tabnet.astype(np.float64)
    x_test_tabnet = x_test_tabnet.astype(np.float64)
    
    # 记录保留的特征名称，确保预测时使用相同的特征集
    tabnet_feature_names = trn_x_tabnet.columns.tolist()
    
    # 验证数据类型和形状
    print(f"TabNet特征数量: {len(tabnet_feature_names)}")
    print(f"训练数据形状: {trn_x_tabnet.shape}, 数据类型: {trn_x_tabnet.dtypes.unique()}")
    print(f"验证数据形状: {val_x_tabnet.shape}")
    print(f"测试数据形状: {x_test_tabnet.shape}")
    
    # 转换为NumPy数组
    X_train_np = trn_x_tabnet.values.astype(np.float32)
    y_train_np = trn_y.values.astype(np.int64)
    X_val_np = val_x_tabnet.values.astype(np.float32)
    y_val_np = val_y.values.astype(np.int64)
    
    # TabNet模型参数
    tabnet_params = {
        "n_d": 64,
        "n_a": 64,
        "n_steps": 5,
        "gamma": 1.5,
        "n_independent": 2,
        "n_shared": 2,
        "cat_idxs": [],
        "cat_dims": [],
        "cat_emb_dim": 1,
        "lambda_sparse": 1e-4,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=2e-2),
        "mask_type": "entmax",
        "scheduler_params": dict(mode="min", patience=5, min_lr=1e-5, factor=0.9),
        "scheduler_fn": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "seed": 2025,
        "verbose": 1
    }
    
    # 添加禁用特征重要性计算的参数
    clf = TabNetClassifier(**tabnet_params)
    
    # 训练TabNet模型
    clf.fit(
        X_train=X_train_np, y_train=y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        compute_importance=False  # 禁用特征重要性计算，避免explain方法被调用
    )
    
    # TabNet预测，确保使用相同的预处理数据和特征
    # 明确使用与训练相同的特征子集
    test_data_for_pred = x_test_tabnet[tabnet_feature_names].values.astype(np.float32)
    val_data_for_pred = val_x_tabnet[tabnet_feature_names].values.astype(np.float32)
    
    tabnet_test_preds = clf.predict_proba(test_data_for_pred)[:, 1]
    tabnet_val_preds = clf.predict_proba(val_data_for_pred)[:, 1]
    
    # 验证预测结果长度
    print(f"验证集原始长度: {len(val_y)}, TabNet预测长度: {len(tabnet_val_preds)}")
    print(f"测试集原始长度: {len(x_test)}, TabNet预测长度: {len(tabnet_test_preds)}")
    
    # 确保长度匹配，如果不匹配则用均值填充或截断
    if len(tabnet_val_preds) != len(val_y):
        print(f"警告：验证集预测长度不匹配，调整长度从 {len(tabnet_val_preds)} 到 {len(val_y)}")
        if len(tabnet_val_preds) < len(val_y):
            # 如果预测结果较短，用均值填充
            mean_pred = np.mean(tabnet_val_preds)
            tabnet_val_preds = np.append(tabnet_val_preds, [mean_pred] * (len(val_y) - len(tabnet_val_preds)))
        else:
            # 如果预测结果较长，截断
            tabnet_val_preds = tabnet_val_preds[:len(val_y)]
    
    if len(tabnet_test_preds) != len(x_test):
        print(f"警告：测试集预测长度不匹配，调整长度从 {len(tabnet_test_preds)} 到 {len(x_test)}")
        if len(tabnet_test_preds) < len(x_test):
            # 如果预测结果较短，用均值填充
            mean_pred = np.mean(tabnet_test_preds)
            tabnet_test_preds = np.append(tabnet_test_preds, [mean_pred] * (len(x_test) - len(tabnet_test_preds)))
        else:
            # 如果预测结果较长，截断
            tabnet_test_preds = tabnet_test_preds[:len(x_test)]
    
    # 创建与x_train长度一致的预测结果数组
    fold_val_preds = np.full(len(x_train), np.nan)
    fold_val_preds[val_idx] = tabnet_val_preds
    
    # 保存TabNet预测结果
    tabnet_pred_y['fold_%d_seed_%d' % (i, seeds)] = tabnet_test_preds
    tabnet_var_pre['fold_%d_seed_%d' % (i, seeds)] = fold_val_preds
    
    # 计算TabNet在验证集上的性能 - 使用调整后的预测结果
    tabnet_score = auc(val_y, tabnet_val_preds)
    tabnet_score_list.append(tabnet_score)
    print(f"TabNet Fold {i+1} AUC: {tabnet_score}")

# 计算两个模型的平均分数
lgb_avg_score = np.mean(lgb_score_list)
tabnet_avg_score = np.mean(tabnet_score_list)
print(f"LightGBM平均AUC: {lgb_avg_score}")
print(f"TabNet平均AUC: {tabnet_avg_score}")

# 根据验证集性能确定模型权重
# 使用验证集上的性能作为权重的基础
lgb_weight = lgb_avg_score / (lgb_avg_score + tabnet_avg_score)
tabnet_weight = tabnet_avg_score / (lgb_avg_score + tabnet_avg_score)

print(f"LightGBM权重: {lgb_weight}, TabNet权重: {tabnet_weight}")

# 模型融合 - 线性加权方法
lgb_preds = pred_y.mean(axis=1).values
tabnet_preds = tabnet_pred_y.mean(axis=1).values

# 融合预测结果
test['is_risk'] = lgb_weight * lgb_preds + tabnet_weight * tabnet_preds

df_test = pd.read_csv('/home/zeriwang/2025Spring/DataMining/2025_Spring_UCAS_DataMining/project/data/evaluation_public.csv')
df_test = pd.merge(df_test,test[['id','is_risk']],how='left')
df_test['op_datetime'] = pd.to_datetime(df_test['op_datetime'])
df_test = df_test.sort_values('op_datetime').reset_index(drop=True)
df_test['hour'] = df_test['op_datetime'].dt.hour

# 根据时间进行后处理调整（深夜和凌晨操作视为高风险）
df_test.loc[df_test['hour']<6,'is_risk'] = 1
df_test.loc[df_test['hour']>20,'is_risk'] = 1

# 保存结果
df_test[['id','is_risk']].to_csv("/home/zeriwang/2025Spring/DataMining/2025_Spring_UCAS_DataMining/project/data/result.csv", index=False)

# 保存单模型结果，便于后续分析
test['lgb_risk'] = lgb_preds
test['tabnet_risk'] = tabnet_preds
test[['id', 'lgb_risk', 'tabnet_risk', 'is_risk']].to_csv(
    "/home/zeriwang/2025Spring/DataMining/2025_Spring_UCAS_DataMining/project/data/model_fusion_detail.csv", index=False)