import os
import numpy as np
import matplotlib
import logging
from PIL import Image
from datetime import datetime
from typing import Union
from pathlib import Path

__all__ = [
    'imnorm',
    'imsave',
    'apply_colormap',
    'np2pil',
    'resize_pil_image',
    'im_dtype2uint8',
    'list2bhwc',
    'Logger'
]

eps = 1e-7


def imnorm(img: np.ndarray, mode=None):
    """normalize image"""
    dtype = img.dtype
    img = img.astype(np.float32)

    if mode is None:
        if dtype == 'uint8':
            img = img / 255.0
        elif dtype == 'uint16':
            img = img / (2 ** 16 - 1)
    elif mode == 'min-max':
        img_min = np.nanmin(img)
        img_max = np.nanmax(img)
        img = (img - img_min) / (img_max - img_min).clip(min=eps)
    elif mode == 'z-score':
        img_mean = img.mean()
        img_std = img.std()
        img = (img - img_mean) / img_std.clip(min=eps)
    else:
        raise ValueError("not supported mode")

    return img


def imsave(image: np.ndarray, save_name: str, save_path: Union[Path, str], cmap_name=None, norm=False):
    """
    :param image: 灰度或彩色图像，np.array 类型，可以是(0,1)范围内的浮点数或(0,255)范围内的整数
    :param save_name: 保存图像的名字
    :param save_path: 保存图像的路径
    :param cmap_name: 伪彩色方案名称，例如'twilight_shifted'
    :param norm: 是否归一化
    """
    assert len(image.shape) in [2, 3]
    if norm:
        image = imnorm(image, mode='min-max')
    else:
        image = imnorm(image)

    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    if cmap_name is not None:
        if len(image.shape) == 3:
            pass
        elif len(image.shape) == 2:
            image = apply_colormap(image, cmap_name)
        else:
            raise ValueError

    uint8_image = (255 * image).astype(np.uint8)
    pil_image = Image.fromarray(uint8_image)

    os.makedirs(save_path, exist_ok=True)
    pil_image.save(os.path.join(save_path, save_name))


def apply_colormap(image: np.ndarray, cmap_name: str):
    """将灰度图转换为伪彩色图"""
    assert image.max() <= 1. and image.min() >= 0.
    assert len(image.shape) == 2

    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    color_image = cmap(image)  # 输出形状为 (H,W,4)（RGBA）或 (H,W,3)（RGB）

    if color_image.shape[-1] == 4:
        color_image = color_image[:, :, :3]

    return color_image


def np2pil(image: np.ndarray):
    uint8_image = im_dtype2uint8(image)
    pil_image = Image.fromarray(uint8_image)
    return pil_image


def resize_pil_image(pil_image: Image.Image, scale_ratio: float):
    return pil_image.resize((int(pil_image.width / scale_ratio), int(pil_image.height / scale_ratio)))


def im_dtype2uint8(image: np.ndarray):
    if image.dtype == np.uint8:
        return image
    elif image.dtype == np.uint16:
        return (image / 65535 * 255).astype(np.uint8)
    elif (image.dtype == np.float32) or (image.dtype == np.float64):
        assert image.min() >= 0 and image.max() <= 1
        return (image * 255).astype(np.uint8)
    else:
        raise TypeError('image dtype must be uint8, uint16 or float32')


def list2bhwc(images: list):
    np_images = np.stack(images)
    if np_images.ndim == 3:
        np_images = np_images[..., np.newaxis]
    elif np_images.ndim == 4:
        np_images = np_images
    else:
        raise ValueError
    return np_images


class Logger:
    def __init__(self, log_path):
        log_name = os.path.basename(log_path)
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log_name = log_name if log_name else "train.log"
        self.log_path = log_path

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(logging.INFO)
        log_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # 配置文件 handler
        file_handler = logging.FileHandler(self.log_path, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

        # 配置屏幕 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(log_formatter)

        # 添加 handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def make_logger(root_dir):
    time_str = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")
    log_dir = os.path.join(root_dir, 'checkpoints', time_str)

    # 创建 logger
    log_path = os.path.join(log_dir, "detail.log")
    logger = Logger(log_path).init_logger()

    return log_dir, logger


