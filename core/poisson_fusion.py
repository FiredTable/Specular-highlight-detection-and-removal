import os
import skimage
import cv2
import numpy as np
import matplotlib.pyplot as plt

from core.possion_blending import create_mask, poisson_blend

__all__ = ['poisson_fusion']


def poisson_fusion(image_src, image_dst, mask, method='normal', visualize=False):
    """
    使用泊松方法融合图像
    :param image_src: 贴图
    :param image_dst: 目标图像
    :param mask: 贴图在目标图像中的位置
    :param method: 'normal', 'mix', 'target', 'src'
    :param visualize: 是否可视化
    """
    assert method in ['normal', 'mix', 'target', 'src']

    # 将 bool 类型的 mask 转化为 uint8 类型
    gray_mask = mask.astype(np.uint8)

    if image_src.ndim == 2:
        color_image_src = np.stack((image_src,) * 3, axis=-1) * 255
        color_image_dst = np.stack((image_dst,) * 3, axis=-1) * 255
    else:
        color_image_src = image_src * 255
        color_image_dst = image_dst * 255

    img_mask, img_src, offset_adj = create_mask(gray_mask.astype(np.float64), color_image_dst,
                                                color_image_src, offset=(0, 0))
    color_blending_image = poisson_blend(img_mask, img_src, color_image_dst, method=method, offset_adj=offset_adj)

    if image_src.ndim == 2:
        gray_blending_image = cv2.cvtColor(color_blending_image, cv2.COLOR_RGB2GRAY)
        blending_image = gray_blending_image.astype(float) / 255.0
    else:
        blending_image = color_blending_image.astype(float) / 255.0

    if visualize:
        plt.figure('possion fusion')

        plt.subplot(141)
        plt.imshow(img_mask)
        plt.title('mask')
        plt.subplot(142)
        plt.imshow(image_src, cmap='viridis', interpolation='None')
        plt.title('image_src')
        plt.subplot(143)
        plt.imshow(image_dst, cmap='viridis', interpolation='None')
        plt.title('image_dst')
        plt.subplot(144)
        plt.imshow(blending_image, cmap='viridis', interpolation='None')
        plt.title('blending_image')

        plt.show(block=True)

    return blending_image


def _test_possion_blend():
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, '../_test_images')
    img1_path = os.path.join(img_dir, 'rho', '00026.png')
    img2_path = os.path.join(img_dir, 'vis', '00022.png')

    img1 = skimage.io.imread(img1_path)
    img2 = skimage.io.imread(img2_path)
    img1 = img1.astype(np.float64) / 255
    img2 = img2.astype(np.float64) / 255

    mask = (img1 > 0.3) & (img2 > 0.3)

    blending_image = poisson_fusion(img1, img2, mask, method='normal', visualize=True)
    plt.imshow(blending_image)