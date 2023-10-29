import torch
import torch.nn as nn
import random
import math
from tqdm import tqdm
import config
from torch.utils.data import DataLoader
from dataset import Training_Dataset, Validation_Dataset


def get_loaders(
        db_root_dir,
        batch_size,
        num_classes=len(config.COCO),
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        rect_training=False,
        box_format="coco",
        ultralytics_loss=False
):

    S = [8, 16, 32]

    train_augmentation = config.TRAIN_TRANSFORMS
    val_augmentation = None

    # bs here is not batch_size, check class method "adaptive_shape" to check behavior
    train_ds = Training_Dataset(root_directory=db_root_dir,
                                transform=train_augmentation, train=True, rect_training=rect_training,
                                bs=batch_size, bboxes_format=box_format, ultralytics_loss=ultralytics_loss)

    val_ds = Validation_Dataset(anchors=config.ANCHORS,
                                root_directory=db_root_dir, transform=val_augmentation,
                                train=False, S=S, rect_training=rect_training, bs=batch_size,
                                bboxes_format=box_format)

    shuffle = False if rect_training else True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        collate_fn=train_ds.collate_fn_ultra if ultralytics_loss else train_ds.collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        collate_fn=None,
    )

    return train_loader, val_loader