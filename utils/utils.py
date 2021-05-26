# 一些常见的处理函数
import torch as t
import torch.nn as nn
import numpy as np
import random
import logging
import csv

def setup_seed(seed):
    '''
    func:设置随机种子
    :param seed:
    :return:
    '''
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


def initial_weight(model):
    net = model
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger



def write_csv(results, file_name):
    # 写入test数据方便后期debug
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)