from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision.transforms import RandomRotation, Normalize, Compose, RandomResizedCrop, ToTensor
import torchvision.transforms as transforms
from transformers import AutoImageProcessor
import os

batch_size = 128
rotate = RandomRotation(degrees=(0, 90))
to_tensor = transforms.ToTensor()
current_folder = os.path.dirname(os.path.abspath(__file__))
cache_data_path = os.path.join(current_folder, 'data')

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

size = (
    image_processor.size["shortest_edge"] if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transform = Compose([RandomResizedCrop(size), ToTensor(), normalize])
def transform(examples):
    examples["image"] = [_transform(image.convert("RGB")) for image in examples["image"]]
    return examples

VAL_RATIO = 0.1
SPLIT_SEED = 1000

def load_imagenet_dataset(dataset_path):
    train_data = load_dataset(dataset_path, split='train', cache_dir=cache_data_path)
    number_label = max(train_data["label"]) + 1
    # the dataset's 'valid' split is used as the test set, so hold out part of train for validation
    split = train_data.train_test_split(test_size=VAL_RATIO, seed=SPLIT_SEED)
    train_data, val_data = split["train"], split["test"]
    train_data.set_transform(transform)
    train_dataset = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                shuffle=True, num_workers=0)
    val_data.set_transform(transform)
    val_dataset = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                shuffle=False, num_workers=0)

    test_data = load_dataset(dataset_path, split='valid', cache_dir=cache_data_path)
    test_data.set_transform(transform)
    test_dataset = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                shuffle=False, num_workers=0)

    image_size = test_data['image'][0].shape[-1]

    return train_dataset, val_dataset, test_dataset, number_label, image_size

def load_pokemon_dataset(dataset_path):
    train_data = load_dataset(dataset_path, 'full', split="train", cache_dir=cache_data_path)
    image_size = train_data['image'][0].size[0]
    number_label = max(train_data["labels"]) + 1
    train_data.set_transform(transform)
    train_dataset = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                shuffle=True, num_workers=0)

    val_data = load_dataset(dataset_path, 'full', split="validation", cache_dir=cache_data_path)
    val_data.set_transform(transform)
    val_dataset = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                shuffle=False, num_workers=0)

    test_data = load_dataset(dataset_path, 'full', split="test", cache_dir=cache_data_path)
    test_data.set_transform(transform)
    test_dataset = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                shuffle=False, num_workers=0)

    return train_dataset, val_dataset, test_dataset, number_label, image_size

