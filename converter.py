import torch
import torch.onnx
import torchvision.models as models
from torch import nn
from torch.autograd import Variable
from vgg import vgg16_bn 
import onnx as nx

def main():
    device = torch.device('cpu')
    model = vgg16_bn()
    PATH = 'weights/imagenet_vgg16_acc_92.096.pt'
    model_state = torch.load(PATH, map_location=device)
    new_model_state = {}
    """ cuda trained models need to be loaded with the nn        
    DataParallel module. prefix incompatibility is caused by
    the module key word
    """  
    """for key in model_state.keys():
        new_model_state[key[7:]] = model_state[key]
        try:
            model.load_state_dict(new_model_state)
        except Exception as e:
            self.logger.exception("Could not load pytorch model")
            return"""
    # convert torch to onnx
    input_size = Variable(torch.randn(1, 3, 224, 224, device='cpu'))
    file_name = PATH.split('/')[1].rstrip('.pt')
    torch.onnx.export(model, input_size, "resources/{0:}.onnx".format(file_name), verbose=True, input_names=["input_1"], output_names=["output_1"])
    onnx_model = nx.load("resources/{0:}.onnx".format(file_name))
    nx.checker.check_model(onnx_model)
    print("model has been verified")

if __name__ == '__main__':
    main()
