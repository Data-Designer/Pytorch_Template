## Pytorch_Template-ZC

### 数据下载

- 从[kaggle比赛官网](https://www.kaggle.com/c/dogs-vs-cats/data) 下载所需的数据
- 解压并把训练集和测试集分别放在一个文件夹中

### 安装

- PyTorch : 可按照[PyTorch官网](http://pytorch.org)的指南，根据自己的平台安装指定的版本
- 安装指定依赖：

```bash
pip install -r requirements.txt
```

### 训练

使用如下命令启动训练：

```
# 在GPU上训练，model指定前需要在model/init文件中添加
python main.py train --train-data-root=data/train --load-model-path=None --lr=0.005 --batch-size=32 --model='ResNet34' --max-epoch=20
```


详细的使用命令 可使用
```bash
python main.py help
```

### 测试

```bash
python main.py --data-root=./data/test --use-gpu=False --batch-size=32
```

### 可视化

```bash
tensorboard --logdir ./logfile/runs --port 8889
```

