# Noise2SamePro
README

## 项目介绍
本项目是基于自监督去噪模型[Noise2Same](https://github.com/divelab/Noise2Same)的改进，通过优化损失函数，提升了模型收敛速度和去噪效果.   

## 使用效果
去噪效果  
![](https://i1.wp.com/img.erpweb.eu.org/imgs/2024/01/e1c64d796c0ed139.jpg)

## 模型特点
### 介绍
本模型依赖于Noise2Same(下称N2S)模型的训练方法，在训练一定次数的N2S上继续训练得到Noise2SamePro(下称N2SP)，使得其收敛速度更快，图片去噪效果更好。  
与N2S模型的对比  
![](https://i1.wp.com/img.erpweb.eu.org/imgs/2024/01/631005200f54fca0.png)    

|          | 方法          |  PSNR  |
| :----:   | :----:        | :----: |
| 传统方法 | Input         | 20.19  |
|          | NLM           | 22.73  |
|  有监督  | Noise2True    | 29.06  |
|          | Noise2Noise   | 28.86  |
|  自监督  | Noise2Same    | 27.09  |
|          | Noise2SamePro | 27.10  |

### 数据集
我们使用了N2S模型提供的[数据集](https://drive.google.com/drive/folders/1VYMo1OoaGxoOLNx6-qIt2Wg03lsZw_kA?usp=sharing),有关数据集构建和数据源的更多详细信息可以在 [Denoising_data](https://github.com/divelab/Noise2Same/blob/main/Denoising_data) 下找到.  

### 检查点
我们提供基于上述数据集训练9万轮次的检查点，请点击[这里](https://drive.google.com/drive/folders/1SL7Bx1TZgj8Ns4BJppA_PalCSY-J4L1R?usp=drive_link)下载.  

## 使用方法
### 环境
```
python==3.7.2
```
- tensorflow==1.15
- scipy
- scikit-image
- tifffile
- gdown
- opencv-python
- numpy
- matplotlib
- PIL

### 目录树
```
Noise2SamePro
├─ test.py *
├─ test_pro.py *
├─ network_configure.py
├─ models.py *
├─ requirements.txt
├─ basic_ops.py
├─ train.py *
├─ train_pro.py *
├─ utils
│  ├─ train_utils.py
│  ├─ evaluation_utils.py
│  └─ predict_utils.py
├─ README.md
├─ network.py
├─ resnet_module.py
├─ Denoising_data ------------------- 默认的数据集文件夹
├─ trained_model -------------------- 默认的模型存放位置
└─ test_single.py

```
您需要关注的是标记有`*`的文件/文件夹，后文会介绍通过修改这些文件进行自定义训练的方法.本项目源代码中同时含有N2S模型与N2SP模型，其中`test.py`,`train.py`是有关N2S模型的训练与测试部分；`test_pro.py`,`train_pro.py`是有关N2SP的训练与测试部分.

下文介绍N2SP的测试与训练方法，N2S与之类似(您也可参照N2S原仓库).  

### 使用
在`test_single.py`中，请修改如下部分进行单张图片的去噪
```python
picture = 'man/' # Adjust path of the picture you want to test
model_dir = 'N2S_PRO' # Adjust your model path
test_single('test_single/' + picture + 'original_image.png', model_dir, 'test_single/' + picture + 'denoised_image.png')
```
执行以下命令并等待即可
```shell
python test_single.py
```
### 测试
如果您使用我们提供的检查点和默认路径，请将其放在`trained_model/`文件夹下(没有则请新建)  
确保`test_pro.py`文件中检查点路径与数据集路径正确.  
```python
model_dir = 'N2S_PRO-90000' # Adjust your model path
data_dir = 'Denoising_data/test/'
model = Noise2Same('trained_models/', model_dir, dim=2, in_channels=1)
```
执行以下命令并等待即可
```shell
python test_pro.py
```

### 训练
N2SP的训练方法是：  

> 对于T次训练，采取`(T-a)`次对于N2S的训练和`a`次对N2SP的训练  

确保路径正确后，修改`train_pro.py`   
```python
model_dir = 'N2S_PRO-8000' # Set model checkpoints save path
steps = 8000 # Set training steps

sgm_loss = 1 # the default sigma is 1
model = Noise2Same('trained_models/', model_dir, dim=2, in_channels=1, lmbd=2*sgm_loss) 
model.train(X[..., None], patch_size=[64, 64], validation=X_val[..., None], batch_size=64, steps=steps-500)
model = Noise2SamePro('trained_models/', model_dir, dim=2, in_channels=1)
model.train(X[..., None], patch_size=[64, 64], validation=X_val[..., None], batch_size=64, steps=steps)

```

在上面的数值中，`steps = 8000`是总步数，`sgm_loss = 1`是N2S模型使用的参数(默认为1),`steps=steps-500`中的500是N2SP的训练次数  
即8000为 上述`T`;500为上述`a`.训练过程是进行7500次N2S的预训练，然后进行500次的附加训练.

调整完毕后，执行以下命令并等待即可
```shell
python train_pro.py
```

### 其他事项

## 贡献者
本项目是天津大学2021图像处理结课作业的去噪部分，由郭骐瑞，刘原驰，罗嘉杰，周欣宇，陈嘉强，李进共同完成