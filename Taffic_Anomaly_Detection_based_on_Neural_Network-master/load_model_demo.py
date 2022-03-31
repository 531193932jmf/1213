import tensorflow as tf
from tensorflow.keras import utils,layers
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

past_history = 400
future_target = 64
STEP = 1
BATCH_SIZE = 128
BUFFER_SIZE = 2000
TRAIN_SPLIT = 40000
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
      labels.append(target[i+target_size])                                      #仅仅预测未来的单个点
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)



model=load_model("D:\pythonProject1\Abnormal_flow\Taffic_Anomaly_Detection_based_on_Neural_Network-master\model_LSTM_IDS_11_20.h5")
print(model.summary())

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
print(dataset.shape)
#标准化
dataset.insert(1,'Label',data_label)
for col in features_considered2:
    scaler = MinMaxScaler()
    dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1,1))
print(dataset.head())

dataset = dataset.values
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 0],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=False)
val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE)
score = r2_score(val_data_single, model.predict(val_data_single))
print("r^2 值为： ", score)








