from transformers import ViTForImageClassification as ViTModel
from transformers import ViTConfig
import torch.nn as nn
# Initializing a ViT vit-base-patch16-224 style configuration

# Initializing a model (with random weights) from the vit-base-patch16-224 style configuration

class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels, image_size):
        super(ViTForImageClassification, self).__init__()
        configuration = ViTConfig(num_labels=num_labels, image_size=image_size, num_hidden_layers=4, num_attention_heads=4)
        configuration.use_bfloat16 = True
        self.vit = ViTModel(configuration)
        self.vit.from_pretrained("google/vit-base-patch16-224-in21k")
        self.num_labels = num_labels
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits