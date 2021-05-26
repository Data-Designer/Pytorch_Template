# -*- coding: utf-8 -*-
# 上传服务器这句话真不能少
import os
import torch as t
import models
import ipdb
import fire
import csv
import datetime
from config import opt
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torch.autograd import Variable
from inspect import getsource
from tqdm import tqdm
from utils.utils import setup_seed,initial_weight,get_logger # 这里引入工具函数
from pytorch_lightning.metrics import Accuracy
from torch.utils.tensorboard import SummaryWriter

import importlib,sys # 编码问题
importlib.reload(sys)

# 常见配置
setup_seed(20) # 设置随机种子
writer = SummaryWriter(opt.tensorboard_log_file) # 定义Tensorboard句柄
print(t.__version__)
logger = get_logger(opt.log_file)


def test(**kwargs):
    # test是没有label的
    logger.info("Start Testing...")
    opt.parse(kwargs) # 更新参数
    ipdb.set_trace()

    # configure model
    model = getattr(models, opt.model)()

    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # data，这里使用训练集测试了，原理上可以在训练集中继续划分
    train_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    model.eval()
    with t.no_grad():
        for step, (data, path) in enumerate(test_dataloader):
            input = Variable(data, volatile=True)
            if opt.use_gpu:
                input = input.cuda()
            score = model(input)
            probability = t.nn.functional.softmax(score)[:, 0].data.tolist()

            batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]

            results += batch_results # 列表相加的结果还是列表
        write_csv(results, opt.result_file) # 记录结果
    logger.info("Finish Testing...")
    return results


def write_csv(results, file_name):
    # 写入test数据方便后期debug
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def train_step(model,features,labels):

    model.train()
    # 梯度清零
    model.optimizer.zero_grad()

    # 正向传播求损失
    predictions = model(features)
    loss = model.criterion(predictions,labels)
    metric = model.metric_func(predictions,labels)

    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()
    model.lr_schedule.step()


    return loss.item(),metric.item()

def valid_step(model,features,labels):

    # 预测模式，dropout层不发生作用
    model.eval()
    # 关闭梯度计算
    with t.no_grad():
        predictions = model(features)
        loss = model.criterion(predictions, labels)
        metric = model.metric_func(predictions, labels)

    return loss.item(), metric.item()


def train(**kwargs):
    opt.parse(kwargs) # 获取命令行更新参数

    # step1: configure model
    model = getattr(models, opt.model)()
    model = initial_weight(model) # 初始化权重
    if opt.load_model_path: # 从最新的checkpoints开始
        model.load(opt.load_model_path) # other_param
    if opt.use_gpu: model.cuda()

    # step2: data
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    # step3: criterion and optimizer
    model.criterion = t.nn.CrossEntropyLoss() # 猴子补丁的写法
    lr = opt.lr
    model.optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    model.lr_schedule = t.optim.lr_scheduler.MultiStepLR(model.optimizer, milestones=[10, 20, 30],
                                                       gamma=0.1)  # update lr, 每10个epoch衰减为原来的0.1

    # step4: metrics
    model.metric_func = Accuracy().cuda()
    model.metric_name = "acc"


    # train
    logger.info("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%nowtime)

    for epoch in range(1, opt.max_epoch+1):

        # loss_meter.reset()
        # confusion_matrix.reset()
        train_loss = 0.0
        metric_sum = 0.0
        step = 1

        for step, (data, label) in tqdm(enumerate(train_dataloader,1), total=len(train_dataloader)):

            # train model
            input = Variable(data) # 转换为Tensor
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            loss,metric = train_step(model,input,target)

            # metric update and visualize
            train_loss += loss
            metric_sum += metric
            writer.add_scalar('train loss', train_loss / step, (epoch-1) * len(train_dataloader) + step)

            # train的时候打印batch日志
            if step % opt.print_freq == opt.print_freq - 1:
                print(("[step = %d] loss: %.3f, " + model.metric_name + ": %.3f") %
                      (step, train_loss / step, metric_sum / step))

                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

        # 可以通过重写save和load函数保存更多参数
        # checkpoint = {
        #     "net": model.state_dict(),
        #     "optimizer": model.optimizer,
        #     "epoch": epoch,
        #     "lr_schedule":model.lr_schedule.state_dict()
        # }
        # model.save(other_param=checkpoint) # 每个epoch保存一次
        model.save()  # 每个epoch保存一次

        # validate and visualize
        val_loss = 0.0
        val_metric_sum = 0.0
        val_step = 1

        with t.no_grad():
            for val_step, (data, label) in enumerate(val_dataloader,1):
                input = Variable(data)  # 转换为Tensor
                target = Variable(label)
                val_input = Variable(input)
                val_label = Variable(target.type(t.LongTensor)) # label需要转成longTensor
                if opt.use_gpu:
                    val_input = val_input.cuda()
                    val_label = val_label.cuda()

                val_loss, val_metric = valid_step(model, val_input, val_label)
                val_loss += val_loss
                val_metric_sum += val_metric
                writer.add_scalar('valid loss', val_loss / step, (epoch-1) * len(val_dataloader) + step)
                writer.add_scalar('valid {}'.format(model.metric_name), val_metric_sum / step, (epoch - 1) * len(val_dataloader) + step)

        info = (epoch, train_loss / step, metric_sum / step,
                val_loss / val_step, val_metric_sum / val_step)

        print(("\nEPOCH = %d, loss = %.3f," + model.metric_name + \
               "  = %.3f, val_loss = %.3f, " + "val_" + model.metric_name + " = %.3f") # 也可以使用log
              % info)
        print('Learning Rate:', model.optimizer.state_dict()['param_groups'][0]['lr'])
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)


    writer.close() # tensorboard --logdir ./logfile/runs --port 8889 转发到丹炉的ai.danlu.netease.com:18540端口
    logger.info("Finish Training!")




def help():
    '''
    打印帮助的信息： python file.py help
    '''

    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    source = (getsource(opt.__class__)).encode(encoding='utf-8')
    print(source)


if __name__ == '__main__':
    fire.Fire() # 命令行运行

