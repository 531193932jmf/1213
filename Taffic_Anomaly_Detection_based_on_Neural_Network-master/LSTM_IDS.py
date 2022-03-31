from __future__ import absolute_import, division, print_function, unicode_literals
import os

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers,regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=2)
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
#use GPU with ID=0
print(tf.test.is_gpu_available())

TRAIN_SPLIT = 40000

#CSV_FILE_PATH = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
#df=pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",header=0,low_memory=False)
#df[" Label"]
#这样就可以了

CSV_FILE_PATH = 'binary_classification.csv'
df = pd.read_csv(CSV_FILE_PATH)
print(len(df))

#Object类型转换为离散数值（Label列）
df['Label'] = pd.Categorical(df['Label'])
df['Label'] = df['Label'].cat.codes
#修改数据类型

columns_counts = df.shape[1]                           #获取列数
for i in range(columns_counts):
  if(df.iloc[:,i].dtypes) != 'float32':
    df.iloc[:, i] = df.iloc[:,i].astype("float32")

#选取11个特征和Label
features_considered = ['Bwd_Packet_Length_Min','Subflow_Fwd_Bytes','Total_Length_of_Fwd_Packets','Fwd_Packet_Length_Mean','Bwd_Packet_Length_Std','Flow_Duration','Flow_IAT_Std','Init_Win_bytes_forward','Bwd_Packets/s',
                 'PSH_Flag_Count','Average_Packet_Size']
features_considered2 = ['Flow_IAT_Std_label','Bwd_Packet_Length_Min','Subflow_Fwd_Bytes','Total_Length_of_Fwd_Packets','Fwd_Packet_Length_Mean','Bwd_Packet_Length_Std','Flow_Duration','Flow_IAT_Std','Init_Win_bytes_forward','Bwd_Packets/s',
                 'PSH_Flag_Count','Average_Packet_Size']
features = df[features_considered]
data_result = df['Flow_IAT_Std']
data_label=df['Label']
data_label=data_label.astype('int')
dataset = features.values
dataset = pd.DataFrame(dataset,columns=features_considered)
dataset['PSH_Flag_Count']=dataset['PSH_Flag_Count'].astype('int')
dataset.insert(0,'Flow_IAT_Std_label',data_result)
dataset.to_csv("data.csv")
print(dataset.shape)
#标准化
dataset.insert(1,'Label',data_label)
for col in features_considered2:
    scaler = MinMaxScaler()
    dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1,1))
print(dataset.head())


dataset = dataset.values
#返回时间窗,根据给定步长对过去的观察进行采样  history_size为过去信息窗口的大小，target_size为模型需要预测的未来时间

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size                                      #如果未指定end_index,则设置最后一个训练点

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])                                      #仅仅预测未来的单个点,可以用于标签分类
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)




past_history = 400
future_target = 64
STEP = 1

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 0], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=False)            #dataset[:,1]取最后一列的所有值


x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 0],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=False)



#训练集、验证集
BATCH_SIZE = 128
BUFFER_SIZE = 2000
train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
model = tf.keras.Sequential([
    layers.LSTM(256,input_shape=x_train_single.shape[-2:],return_sequences=True),#注意输入维度
    layers.Dropout(0.2),
    layers.LSTM(128,return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(64),
    #一会试一下 预测。像天气预测一样。那么不要再有激活函数。
    layers.Dense(128, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)),#这个不是batch_size，而是future_target，值一样而已。
    layers.Dense(64)
])




print(model.summary())


model.compile(optimizer='adam',
              loss='mse',)
log_dir = "graph/log_fit/7"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_path = "traing_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# latest=tf.train.latest_checkpoint(checkpoint_dir)
# 创建一个回调，每 1个 epochs 保存模型.
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 verbose=1,
                                                 save_weights_only=True,
                                                 save_freq=200)
# checkpoint_file = "best_model.hdf5"
#
# checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file,
#                                       monitor='loss',
#                                       mode='min',
#                                       save_best_only=True,
#                                       save_weights_only=True)#没有用到而已
model.save_weights(checkpoint_path.format(epoch=0))
# model.load_weights(latest)
history=model.fit(train_data_single,epochs=5,steps_per_epoch=200,batch_size=128,
                  callbacks=[cp_callback,early_stopping,tensorboard_callback])
plt.figure(figsize=(16,8))
plt.plot(history.history['loss'], label='train loss')
plt.legend(loc='best')
plt.show()
print(model.output_shape)
print(model.summary())
model.save("model_LSTM_IDS_11_20.h5")

score = r2_score(val_data_single, model.predict(val_data_single))
print("r^2 值为： ", score)

def multi_step_plot(history,true_future,prediction):
    plt.figure(figsize=(12,6))
    num_in=create_time_steps(len(history))
    num_out=len(true_future)

    plt.plot(num_in,np.array(history[:,1]),label='History')

    plt.plot(np.arange(num_out)/STEP,np.array(true_future),'bo',label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP,np.array(prediction),'ro',label='Predicted future')
    plt.legend(loc='upper left')
    plt.show()
#建立时间序列，为了展示
def create_time_steps(length):
#建立时间序列，为了展示
    time_steps=[]
    for i in range(-length,0,1):
        time_steps.append(i)
    return time_steps
reconstructed_model = tf.keras.models.load_model('LSTM_Model_11_20.h5')


for x,y in val_data_single.take(3):#拿出来三个 batch_size
    print("model.predict(x).shape==================",model.predict(x).shape)
    #第一个256为batch_size,
    print("x:\n",x[0][:, 1].numpy(), "y:\n",y[0].numpy(),"pre:\n",model.predict(x)[0])
    result=model.predict(x)
    plt.subplot(2,2,1)
    multi_step_plot(x[0],y[0],result[0])
    plt.subplot(2, 2, 2)
    multi_step_plot(x[1], y[1], result[1])
    plt.subplot(2, 2, 3)
    multi_step_plot(x[2], y[2], result[2])
    plt.subplot(2, 2, 4)
    multi_step_plot(x[2], y[2], result[2])
    print("result:",np.argmax(result))
