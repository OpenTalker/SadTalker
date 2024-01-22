#%%
# pip install onnxconverter_common onnx onnxruntime-tools
# pip install onnxruntime-gpu
# gpt环境
#%%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
        self.weight = nn.Parameter(torch.rand(1).float()).float()

    def forward(self, x:torch.FloatTensor):
        
        return x + self.weight
    
model = Add()
x = torch.rand((1, 2)).float()
print(model(x))

import onnxruntime as ort
# https://zhuanlan.zhihu.com/p/422290231#:~:text=%E5%9C%A8pytorch%E4%B8%AD%E8%BD%AC%E6%8D%A2%E4%B8%BAonnx%E7%9A%84%E6%A8%A1%E5%9E%8B%EF%BC%8C%E5%B0%B1%E4%B8%80%E8%A1%8C%E4%BB%A3%E7%A0%81%EF%BC%9A%20torch.onnx.export%28model%2C%20args%2C%20f%2C%20export_params%3DTrue%2C%20verbose%3DFalse%2C%20input_names%3DNone%2C,output_names%3DNone%2Cdo_constant_folding%3DTrue%2Cdynamic_axes%3DNone%2Copset_version%3D9%29%20%E5%B8%B8%E7%94%A8%E5%8F%82%E6%95%B0%EF%BC%9A%201.model%3Atorch.nn.model%20%E8%A6%81%E5%AF%BC%E5%87%BA%E7%9A%84%E6%A8%A1%E5%9E%8B%202.args%3Atuple%20or%20tensor%20%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%BE%93%E5%85%A5%E5%8F%82%E6%95%B0%E3%80%82

import tempfile


class TorchCompile(nn.Module):
    def __init__(self, model:nn.Module, 
                 file_path=None, 
                 exist_recompile=True) -> None:
        super().__init__()
        self.session = None
        
        if file_path is None:
            file_path = tempfile.NamedTemporaryFile(prefix='MyTorchCompile', suffix='.onnx').name
        elif not exist_recompile and os.path.isfile(file_path):
            try:
                self.session = ort.InferenceSession(file_path, 
                                            providers=['CUDAExecutionProvider'])
            except Exception as e:
                print(f'Cannot load model from {file_path} due to "{e}", will recompile later.')
                self.session = None
        self.model = model
        self.file_path = file_path
        
    def init_session_from_model(self, input_tensor):
        print(f"First time calling function {self.model}, compiling for you!")
        # arg_size = tuple(input_tensor.size())
        
        # print(f"Input size infered to be {arg_size}")
        torch.onnx.export(self.model, 
                        #   arg_size, 
                            input_tensor, 
                            self.file_path, 
        export_params=True, verbose=True, input_names=['input'], 
            output_names=['output'], do_constant_folding=True,
            # dynamic_axes=None, 
            dynamic_axes= {
                'input': {
                    0: 'batch',
                },
                'output': {
                    0: 'batch'
                }
            }, 

            # opset_version=9
            opset_version=11
            )
        from onnxruntime_tools import optimizer
        import onnx
        # https://zhuanlan.zhihu.com/p/459875044
        # https://github.com/kenwaytis/faster-SadTalker-API/commit/3bd28ea49e2a48c4b46d9bfe8b693f660fd38cf2
        try:
            # original_model = onnx.load(self.file_path)
            # passes = ['fuse_bn_into_conv']
            # onnx.save(optimized_model, self.file_path)
            
            optimized_model = optimizer.optimize_model(self.file_path, 
            # self.file_path = optimizer.optimize_by_onnxruntime(self.file_path, 
                                                        # optimized_model_path=self.file_path, 
                                                        opt_level=99,
                                                        use_gpu=True,
                                                        )
            optimized_model.convert_model_float32_to_float16()
            optimized_model.save_model_to_file(self.file_path)
            # onnx.save(optimized_model, self.file_path)
            
        except Exception as e:
            raise e
            print(f'due to "{e}", we skip onnx optimize.')
        
        self.session = ort.InferenceSession(self.file_path, 
                                            providers=['CUDAExecutionProvider'])
        for var in self.session.get_inputs():
            print(var.name)
            print(var.shape)
            print(var.type)
        for var in self.session.get_outputs():
            print(var.name)
            print(var.shape)
            print(var.type)
            
            
        print(f"Compiled {self.model} to {self.file_path}!")
    def forward(self, input_tensor):
        if self.session is None:
            self.init_session_from_model(input_tensor)
        # return model(input_tensor)
        outputs = self.session.run(None, {
            # "input"
            self.session.get_inputs()[0].name
                                          :input_tensor.detach().cpu().numpy()})
        return torch.tensor(outputs[0], device=input_tensor.device)
        
model = TorchCompile(model, 'test.onnx')
# model = TorchCompile(model)
model(x)

# 这两个值完全不一样，继续看onnx和onnxruntime的文档
# %%
# x = torch.Tensor([1,2,3])
# x = torch.rand((3, 2)).cuda()
x = torch.rand((3, 2))
model(x)
# %%
