#%%
# pip install onnxruntime-gpu onnx
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
        self.weight = nn.Parameter(torch.rand(1))

    def forward(self, input):
        return input + self.weight
    
model = Add()
x = torch.rand(1)
model(x)
import onnxruntime as ort
# https://zhuanlan.zhihu.com/p/422290231#:~:text=%E5%9C%A8pytorch%E4%B8%AD%E8%BD%AC%E6%8D%A2%E4%B8%BAonnx%E7%9A%84%E6%A8%A1%E5%9E%8B%EF%BC%8C%E5%B0%B1%E4%B8%80%E8%A1%8C%E4%BB%A3%E7%A0%81%EF%BC%9A%20torch.onnx.export%28model%2C%20args%2C%20f%2C%20export_params%3DTrue%2C%20verbose%3DFalse%2C%20input_names%3DNone%2C,output_names%3DNone%2Cdo_constant_folding%3DTrue%2Cdynamic_axes%3DNone%2Copset_version%3D9%29%20%E5%B8%B8%E7%94%A8%E5%8F%82%E6%95%B0%EF%BC%9A%201.model%3Atorch.nn.model%20%E8%A6%81%E5%AF%BC%E5%87%BA%E7%9A%84%E6%A8%A1%E5%9E%8B%202.args%3Atuple%20or%20tensor%20%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%BE%93%E5%85%A5%E5%8F%82%E6%95%B0%E3%80%82
def torch_compile(model:nn.Module, file_path='test.onnx'):
    
    # onnx = torch.onnx.load(name)
    first_call = True
    def wrapped(input_tensor):
        nonlocal first_call
        session = None
        if first_call:
            print(f"First time calling function {model}, compiling for you!")
            first_call = False
            arg_size = tuple(input_tensor.size())
            print(f"Input size is {arg_size}")
            torch.onnx.export(model, 
                              arg_size, 
                              file_path, 
            export_params=True, verbose=True, input_names=['input'], 
                output_names=None, do_constant_folding=True,
                dynamic_axes=None,opset_version=9)
            session = ort.InferenceSession(file_path)
            print(f"Compiled {model} to {file_path}!")
        # return model(input_tensor)
        outputs = session.run(None, {"input":input_tensor.long().numpy()})
        return outputs
        
        
    return wrapped
model = torch_compile(model)
model(x)
# %%
