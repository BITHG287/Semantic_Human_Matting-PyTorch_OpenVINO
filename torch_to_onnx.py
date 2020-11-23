import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
import onnx


if __name__ == "__main__":

    x = torch.randn(1, 3, 256, 256, device='cpu')
    model = torch.load("./ckpt/human_matting/model/model_obj.pth", map_location=torch.device('cpu'))
    model.eval()
    output = model(x)
    model.cpu()
    torch.onnx.export(model, x, "model_obj.onnx", output_names={"output"}, verbose=True)
    onnx_model = onnx.load("model_obj.onnx")
    onnx.checker.check_model(onnx_model)

