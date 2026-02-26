#!/usr/bin/env python3

import torch.nn as nn
from torchvision.models import Weights, WeightsEnum
from torchvision.models import list_models, get_model_builder, get_model, get_model_weights
from torchvision.models import MobileNet_V3_Small_Weights, MobileNetV3


def list_model_weights(name:str) -> list[WeightsEnum]:
  return list(get_model_weights(name))


def get_pretrained_model(name:str, weights:WeightsEnum='auto') -> nn.Module:
  if weights == 'auto':
    weights = get_model_weights(name)['DEFAULT']
  return get_model(name, weights=weights)


def get_mbv3_small(num_classes:int=10) -> MobileNetV3:
  model: MobileNetV3 = get_pretrained_model('mobilenet_v3_small', MobileNet_V3_Small_Weights.IMAGENET1K_V1)
  linear_old: nn.Linear = model.classifier[-1]
  linear_new = nn.Linear(linear_old.in_features, num_classes, bias=linear_old.bias is not None)
  model.classifier[-1] = linear_new
  return model
