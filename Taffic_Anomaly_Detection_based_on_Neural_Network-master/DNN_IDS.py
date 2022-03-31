#!/usr/bin/env python
# coding=UTF-8  
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #显示信息等级
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.test.is_gpu_available())
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import datetime

start_time = datetime.datetime.now()

CSV_FILE_PATH = 'binary_classification.csv'
#四个类

df = pd.read_csv(CSV_FILE_PATH)

#Object类型转换为离散数值（Label列）
df['Label'] = pd.Categorical(df['Label'])
df['Label'] = df['Label'].cat.codes
print(df['Label'].head(20))
#将int类型转换为float类型
columns_counts = df.shape[1]                                                #获取列数
for i in range(columns_counts):
  if(df.iloc[:,i].dtypes) != 'float64':
    df.iloc[:, i] = df.iloc[:,i].astype(float)


#将特征随时间变化用图像展示出来


# ts = df['Init_Win_bytes_forward']
# ts.plot(title='PortScan:Init_Win_bytes_forward')
# plt.xlabel('Time-Step')
# plt.ylim(0,80000)
# plt.xlim(0,1000)
# plt.show()




#选取11个特征和Label
features_considered = ['Label','Bwd_Packet_Length_Min','Subflow_Fwd_Bytes','Total_Length_of_Fwd_Packets','Fwd_Packet_Length_Mean','Bwd_Packet_Length_Std','Flow_Duration','Flow_IAT_Std','Init_Win_bytes_forward','Bwd_Packets/s',
                 'PSH_Flag_Count','Average_Packet_Size']
feature_last = ['Bwd_Packet_Length_Min','Subflow_Fwd_Bytes','Total_Length_of_Fwd_Packets','Fwd_Packet_Length_Mean','Bwd_Packet_Length_Std','Flow_Duration','Flow_IAT_Std','Init_Win_bytes_forward','Bwd_Packets/s',
                 'PSH_Flag_Count','Average_Packet_Size']
feature = df[features_considered]

print(len(feature))

#将数据集分为训练集、验证集、测试集
train, test = train_test_split(feature,test_size=0.2)
print("train_shape",train.shape)

print("test_shape",test.shape)

#标准化
def normalize_dataset(dataset, dataset_mean, dataset_std, insert_target):
    dataset = (dataset-dataset_mean)/dataset_std
    final_dataset = pd.DataFrame(dataset, columns=feature_last)
    final_dataset.insert(0, 'Label', insert_target)
    return final_dataset

train.reset_index(drop=True,inplace=True)
#重置索引，很关键！ drop=True,把原来的索引index列丢掉
train_target = train['Label']
print("Target_size",train_target.shape)
train_other = train[feature_last]
train_dataset = train_other.values
train_mean = train_dataset.mean(axis=0)
train_std = train_dataset.std(axis=0)
train = normalize_dataset(train_dataset, train_mean, train_std, train_target)

#对测试集进行标准化时使用训练集的均值和标准差
test.reset_index(drop=True,inplace=True)
test_target = test['Label']
test_other = test[feature_last]
test_dataset = test_other.values
test = normalize_dataset(test_dataset, train_mean, train_std, test_target)

train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

#使用tf.data.Dataset读取数据
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Label')
  #如果赋值给变量，返回的是Series
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

#选择要使用的
feature_use = []
for header in feature_last:
  feature_use.append(feature_column.numeric_column(header))
print(len(feature_use),len(feature_use[0]),feature_use[0])
print("featuer_use"+"="*20+"\n")

feature_layer = tf.keras.layers.DenseFeatures(feature_use,dtype='float64')

batch_size = 50                                                             #50-256

print("train_shape..........",train.shape)
train_ds = df_to_dataset(train, batch_size=batch_size)
# print(list(train_ds.as_numpy_iterator())[0])
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
print("train_ds.take(1).....",train_ds.take(1))

#创建，编译和训练模型
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(256, activation='relu'),
  layers.Dense(128, activation='relu'),
layers.Dropout(0.2),
  layers.Dense(64, activation='relu'),
layers.Dropout(0.2),
  layers.Dense(32, activation='relu'),
  layers.Dense(4, activation='softmax')
])
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)


#其他指标、使用Tensorboard，

recall=tf.keras.metrics.Recall(),
FP=tf.keras.metrics.FalsePositives(),
TN=tf.keras.metrics.TrueNegatives()

log_dir = "graph/log_fit/13"
checkpoint_path = "traing_1/cp-{epoch:04d}.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 verbose=1,
                                                 save_weights_only=True,
                                                 save_freq=200)


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# model.save_weights(checkpoint_path.format(epoch=0))
history=model.fit(train_ds,
          validation_data=val_ds,
          epochs=25,callbacks=[tensorboard_callback,cp_callback]
          )

loss, accuracy = model.evaluate(test_ds)



classifer=["BENIGN","DDoS","Hulk","PortScan"]


plt.figure(figsize=(16,8))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(loc='best')
plt.show()

for x, y in test_ds.take(3):  # 拿出来3个 batch_size
    print("model.predict(x).shape==================", model.predict(x).shape)
# 第一个256为batch_size,但也不一定，看看批量处理的x是多大。
    for i in range(5):
        result = model.predict(x)[i]  # 得到的是4个的概率
        print("pre_result:", np.argmax(result),"true_result",y[i])
        print("pre_result:", classifer[np.argmax(result)], "true_result", classifer[int(y[i])])



def result_csv(val_data_single):
    j=1;
    result_df=pd.DataFrame(columns=["x","y","pre"])
    for x, y in val_data_single.take(3):  # 拿出来3个 batch_size
        print("model.predict(x).shape==================", model.predict(x).shape)
        for i in range(50):
        # 第一个256为batch_size,但也不一定，看看批量处理的x是多大。
            result = model.predict(x)[0]#得到的是4个的概率
            print("result:", np.argmax(result))
            result=model.predict(x)[i]
            result=np.argmax(result)
            result_df.loc[j, :] = [j, y[i].numpy(),result]
            j+=1
    result_df.to_csv("result_DNN.csv",header=False,index=False)
# result_csv(test_ds)#暂时不需要

model.save_weights("dnn_model")
end_time = datetime.datetime.now()
print("spend_time:",(end_time-start_time).seconds)
#保存模型
begin_time = datetime.datetime.now()
# reconstructed_model = tf.keras.models.load_model('DNN_Model.h5')
# reconstructed_model.evaluate(test_ds)
final_moment = datetime.datetime.now()
print('保存的模型预测时间：', (final_moment-begin_time).seconds)
print("loss:",loss)
print("Accuracy:", accuracy)
print(model.summary())

