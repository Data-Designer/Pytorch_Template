import torch as t
import time
from config import opt

class BasicModule(t.nn.Module):
    """封装nn.Module，提供save和load两种方法"""

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self)) # 默认名字

    def load(self,path):
        """
        func:加载指定路径的模型参数和模型
        :param path:
        :return:
        """
        self.load_state_dict(t.load(path))

    def save(self,name=None):
        """
        func：保存模型，默认使用“模型名字+时间作为文件名”
        :param name:
        :return:
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(),name)
        return name


class Flat(t.nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)