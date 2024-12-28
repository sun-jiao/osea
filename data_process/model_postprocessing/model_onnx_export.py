import sys

import torch
from torchvision.models import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet34(num_classes=11000)
weights_path = sys.argv[1]
onnx_path = sys.argv[2]

model.load_state_dict(torch.load(weights_path))
model = model.to(device)
model.eval()

example = torch.randn(1, 3, 224, 224).to(device)

onnx_program = torch.onnx.dynamo_export(model, example)

onnx_program.save(onnx_path)
