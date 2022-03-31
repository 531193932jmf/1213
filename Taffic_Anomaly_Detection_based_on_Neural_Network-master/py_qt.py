import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtGui import QIcon


################################################
#######创建主窗口
################################################
result = pd.read_csv("result_D.csv", header=None)
result.columns = ["x", "y", "pre"]
result.astype("int64")

def multi_step_plot(true_future,prediction):
    plt.figure(figsize=(12,6))
    num_y=len(true_future)
    num_p=len(prediction)
    plt.plot(num_y,true_future,'bo',label="true classfier")
    if prediction.any():
        plt.plot(num_p,prediction,'ro',label='Predicted future')
    plt.imsave

class FirstMainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('异常检测')

        ###### 创建界面 ######
        self.centralwidget = QWidget()
        self.setCentralWidget(self.centralwidget)
        self.Layout = QVBoxLayout(self.centralwidget)

        # 设置顶部三个按钮
        self.topwidget = QWidget()
        self.Layout.addWidget(self.topwidget)
        self.buttonLayout = QHBoxLayout(self.topwidget)

        self.pushButton1 = QPushButton()
        self.pushButton1.setText("预测1")
        self.buttonLayout.addWidget(self.pushButton1)

        self.pushButton2 = QPushButton()
        self.pushButton2.setText("预测2")
        self.buttonLayout.addWidget(self.pushButton2)

        self.pushButton3 = QPushButton()
        self.pushButton3.setText("敬请期待")
        self.buttonLayout.addWidget(self.pushButton3)
        self.setWindowIcon(QIcon('image/123.gif'))
        # 设置中间文本
        self.edit = QLineEdit()
        self.edit.setPlaceholderText('预测序号:有最大值哦！')
        self.Layout.addWidget(self.edit)
        self.label = QLabel()
        self.label.setText("小组成员:\n 靳明飞,\n李林,\n方涓,\n李斯敏,")
        self.label.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Roman times", 20, QFont.Bold))
        self.Layout.addWidget(self.label)
        # 设置状态栏
        self.statusBar().showMessage("当前用户：靳明飞")
        # 窗口最大化
        #self.showMaximized()

        ###### 三个按钮事件 ######
        self.pushButton1.clicked.connect(self.on_pushButton1_clicked)
        self.pushButton2.clicked.connect(self.on_pushButton2_clicked)
        self.pushButton3.clicked.connect(self.on_pushButton3_clicked)
        self.edit.returnPressed.connect(self.on_pushButton1_clicked)

    # 按钮一：打开主界面
    windowList = []
    def on_pushButton1_clicked(self):
        #预处理操作:
        text = the_mainwindow.edit.text()
        data = result.head()
        print(data)
        labels = ["BENIGN", "DDoS", "DoS_Hulk", "PostScan-real"]
        if len(text)!=0:
            print(type(text))
            try:
                print(result[result['x']==int(text)])
                da=result[result['x']==int(text)]
                da=da.astype("int64")
                if da.empty:
                    print("对不起，没找到")
                    show_data=""
                else:
                    show_data="首列为序号\n"+"\n"+"预测类别:"+str(da['pre'])
                    label=labels[(da['pre'].values[0])]
                    show_data=show_data+"\n预测类别名称:"+label
                    print(show_data)
            except:
                QMessageBox.information(self,"这不是一个数字")

            #y = result.loc[text, ['y']]
            #pre = result.loc[text, ['pre']]
            print(text)
        else:
            show_data=""
        #调用第二界面:
        the_window =SecondWindow(show_data)
        self.windowList.append(the_window)   ##注：没有这句，是不打开另一个主界面的！
        self.close()
        the_window.show()
    # 按钮二：
    def on_pushButton2_clicked(self):
        the_window = ThirdWindow(0)
        self.windowList.append(the_window)  ##注：没有这句，是不打开另一个主界面的！
        self.close()
        the_window.show()


    # 按钮三：打开提示框
    def on_pushButton3_clicked(self):
        QMessageBox.information(self, "提示", "会加油的！！！")
        #QMessageBox.question(self, "提示", "这是question框！")
        #QMessageBox.warning(self, "提示", "这是warning框！")
        #QMessageBox.about(self, "提示", "这是about框！")

################################################
#######第二个主界面
################################################
class SecondWindow(QMainWindow):
    def __init__(self,show_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('靳明飞......')
        self.setWindowIcon(QIcon('image/123.gif'))
        # 设置中间文本
        self.label = QLabel()
        if len(show_data)==0:
            self.label.setText("什么都没有")
            self.setFixedSize(640, 480)
        else:
            self.label.setText(show_data)
        self.label.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Roman times", 25, QFont.Bold))
        self.setCentralWidget(self.label)

        # 设置状态栏
        self.statusBar().showMessage("当前用户：靳明飞")

        # 窗口最大化
        #self.showMaximized()


    ###### 重写关闭事件，回到第一界面
    windowList = []
    def closeEvent(self, event):
        the_window =the_mainwindow
        self.windowList.append(the_window)  ##注：没有这句，是不打开另一个主界面的！
        the_window.show()
        event.accept()

class ThirdWindow(QMainWindow):
    def __init__(self,show_picture, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('靳明飞')
        self.centralwidget = QWidget()
        self.setCentralWidget(self.centralwidget)
        self.Layout = QVBoxLayout(self.centralwidget)

        self.topwidget = QWidget()
        self.Layout.addWidget(self.topwidget)
        self.buttonLayout = QHBoxLayout(self.topwidget)
        self.lab1 = QLabel("前100的预测情况")
        self.lab1.setFont(QFont("Roman times", 15, QFont.Bold))
        self.buttonLayout.addWidget(self.lab1)
        # 设置中间文本

        self.label = QLabel()
        self.label.setPixmap(QPixmap("prediction_true.png"))
        self.Layout.addWidget(self.label)
        self.label2 = QLabel()
        self.label2.setPixmap(QPixmap("prediction_true2.png"))
        self.Layout.addWidget(self.label2)

        # 设置状态栏
        self.statusBar().showMessage("当前用户：靳明飞")

        # 窗口最大化
        #self.showMaximized()


    ###### 重写关闭事件，回到第一界面
    windowList = []
    def closeEvent(self, event):
        the_window =the_mainwindow
        self.windowList.append(the_window)  ##注：没有这句，是不打开另一个主界面的！
        the_window.show()
        event.accept()

################################################
#######对话框
################################################
class TestdemoDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('提示框')
        ### 设置对话框类型
        self.setWindowFlags(Qt.Tool)
################################################
#######程序入门
################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    the_mainwindow = FirstMainWindow()
    the_mainwindow.show()
    sys.exit(app.exec_())
