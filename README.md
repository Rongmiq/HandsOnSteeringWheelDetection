<p align="center">
<img align="center" src="/logo.png", width=1600>
<p>
  
## Hands On Steering Wheel Detection with PaddlePaddle and AI Studio
### 一、项目背景介绍
目前市面上的自动驾驶，均采用智能辅助+人工监督的方式。如果脱离了人为的监督和干涉就容易导致危险的发生，为了有效提高驾驶安全性，方向盘离手检测被当作是汽车自动驾驶系统的必备选项。通过检测驾驶员的手是否离开方向盘而进行提醒警示，从而保障行车安全。已经实现了的方向盘离手检测的方法有多种，比如电容感应方式、图像识别方式、压力感应方式等。我们基于图像识别的方法设计了一个方向盘离手检测预警系统，具有成本低、易实现，精度高等优点。百度飞桨的AI开发套件简化了项目的开发流程，确保了项目的精度和速度。
  
### 二、数据介绍
HandsOnSteeringWheel  VOC格式的数据集，只标注了手这一个类别 https://aistudio.baidu.com/aistudio/datasetdetail/69849
<div>
<ul>
<li><img src="/1_0000086_0_0_0_6.png" /></li>
<li><img src="/10_0000739_0_0_0_0.png" /></li>
<li><img src="/11_0000332_0_0_0_0.png"/></li>
</ul>
</div>  

### 三、模型介绍
目标检测算法的准确性和推理速度不可兼得，为了得到一个兼具性能和速度的检测器，我们选择了比yolo3基础模型训练更快，准确率更高的模型：PP-YOLO 
paper:https://arxiv.org/abs/2007.12099 
Github:https://github.com/PaddlePaddle/PaddleDetection。

  
### 1.Installation paddlex
pip install "paddlex<=2.0.0" -i https://mirror.baidu.com/pypi/simple


### 3.models
 
 
