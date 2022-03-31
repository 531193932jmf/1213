import pandas as pad
import random as ran
import matplotlib.pyplot as plt
pd=pad.DataFrame(columns=["x","y","pre"])
print(pd)
ran.seed(0)
pd.loc[1,:]=[1,2,3]
print(pd)
pd.loc[2,:]=[4,5,6]
print(pd)
for i in range(10):
    pd.loc[i+3,:]=[ran.randint(1,10),ran.randint(1,10),ran.randint(1,10)]
print(pd)
pd.to_csv("shiyan.csv",header=False,index=False)
labels = ["BENIGN", "DDoS", "DoS_Hulk", "PostScan-real"]
da= pd[pd['x']==1]
result=da
show_data = "所属类别:" + str(da['y']) + "\n" + "预测类别:" + str(da['pre'])
print(labels[da['pre'].values[0]])
print(show_data)
if result.empty:
    print("空了")
else:print(result)
def multi_step_plot(true_future,prediction):
    plt.figure(figsize=(6,3))
    num_y=len(true_future)
    num_p=len(prediction)
    plt.plot(range(num_y),true_future,'bo',label="true classfier")
    if prediction.any():
        plt.plot(range(num_p),prediction,'ro',label='Predicted future')
    plt.savefig('prediction_true2.png')
    plt.title('blue is true')
    plt.show()
result = pad.read_csv("result_D.csv", header=None)
result.columns = ["x", "y", "pre"]
print(len(result['y'].values[:100]))
print(len(result['pre'].values[:100]))
multi_step_plot(result['y'].values[:100],result['pre'].values[:100])

