from .AlexNet import AlexNet
from .ResNet34 import ResNet34 # 主类名
# from torchvision.models import InceptinV3
# from torchvision.models import alexnet as AlexNet

'''
写完module后可以在这里直接添加,这样主函数调用起来很方便
from models import AlexNet
或
import models
model = models.AlexNet()
或
import models
model = getattr('models', 'AlexNet')()
'''