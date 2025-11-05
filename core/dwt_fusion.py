import pywt
import numpy as np
import matplotlib.pyplot as plt

from core.common import imnorm

__all__ = ['DWTFuser']


class DWTFuser:
    def __init__(self, caf_weight: list, cxf_weight: list, level: int = 1, wavelet='db10',
                 caf_type='mean', cxf_type='mean'):
        """基于离散小波变换的偏振特征图像融合
        :param caf_weight: 低频分量的融合权重
        :param cxf_weight: 高频分量的融合权重
        :param level: 小波分解的层数
        :param wavelet: 小波基:
            Daubechies 小波（dbN）: 'db1', 'db2', 'db3', ..., 'db40',
            Symlets 小波（symN）: 'sym1', 'sym2', 'sym3', ..., 'sym20'
            Coiflets 小波（coifN）: 'coif1', 'coif2', ..., 'coif5'
        :param caf_type: 分解低频系数的融合方式，可以设置为'min','max','mean'
        :param cxf_type: 分解细节系数的融合方式，可以设置为'min','max','mean'
        """
        self.caf_weight = caf_weight
        self.cxf_weight = cxf_weight
        self.level = level
        self.wavelet = wavelet
        self.caf_type = caf_type
        self.cxf_type = cxf_type

        self.caf_weight = self.normalize_weight(self.caf_weight)
        self.cxf_weight = self.normalize_weight(self.cxf_weight)

    @staticmethod
    def bhwc_imnorm(images: np.ndarray, norm_mode=None):
        normalized_images = np.zeros_like(images, dtype=np.float32)
        for k in range(images.shape[0]):
            normalized_images[k, :] = imnorm(images[k, :], mode=norm_mode)
        return normalized_images

    @staticmethod
    def normalize_weight(weight):
        """对权重进行归一化"""
        weight = np.array(weight, dtype=np.float32)
        weight_norm = weight / np.sum(weight)
        return weight_norm.tolist()

    def fusion(self, images: np.ndarray, norm_mode=None, visualize=False):
        images = self.bhwc_imnorm(images, norm_mode=norm_mode)
        batch_size, height, width, channels = images.shape

        fused_image = np.zeros((height, width, channels), dtype=np.float32)
        for c in range(channels):
            coeffs = []  # 存储所有图像的系数
            for b in range(batch_size):
                # [cAn, (cHn, cVn, cDn), ...(cH1, cV1, cD1)]: list
                coeffs.append(pywt.wavedec2(images[b, :, :, c], self.wavelet, level=self.level))

            # 初始化融合后的系数
            fused_coeff = []
            for l in range(self.level + 1):
                if l == 0:
                    # 处理近似系数 (cA)
                    ca_list = [coeff[l] for coeff in coeffs]  # 提取所有图像的 cA
                    ca_array = np.stack(ca_list, axis=0)  # 转换为 [batch_size, h, w]
                    fused_ca = self._coeff_fusion(ca_array, self.caf_weight, self.caf_type)
                    fused_coeff.append(fused_ca)
                else:
                    # 处理细节系数 (cH, cV, cD)
                    ch_list = [coeff[l][0] for coeff in coeffs]  # 提取所有图像的 cH
                    cv_list = [coeff[l][1] for coeff in coeffs]  # 提取所有图像的 cV
                    cd_list = [coeff[l][2] for coeff in coeffs]  # 提取所有图像的 cD

                    # 转换为 [batch_size, h, w]
                    ch_array = np.stack(ch_list, axis=0)
                    cv_array = np.stack(cv_list, axis=0)
                    cd_array = np.stack(cd_list, axis=0)

                    # 融合细节系数
                    fused_ch = self._coeff_fusion(ch_array, self.cxf_weight, self.cxf_type)
                    fused_cv = self._coeff_fusion(cv_array, self.cxf_weight, self.cxf_type)
                    fused_cd = self._coeff_fusion(cd_array, self.cxf_weight, self.cxf_type)

                    # 将融合后的细节系数添加到融合系数列表中
                    fused_coeff.append((fused_ch, fused_cv, fused_cd))

            # 重建融合后的图像
            fused_image[:, :, c] = pywt.waverec2(fused_coeff, self.wavelet)

            if visualize:
                self._visualize_fusion(fused_coeff, fused_image[:, :, c], self.level)

        if fused_image.shape != (height, width, channels):
            fused_image = fused_image[0: height, 0: width, :]

        if channels == 1:
            fused_image = fused_image[:, :, 0]

        return fused_image

    @staticmethod
    def _coeff_fusion(coeff, weight, fusion_type):
        """分解系数融合"""
        batch_size, height, width = coeff.shape
        if fusion_type == 'mean':
            fused_coeff = np.zeros((height, width), dtype=np.float32)
            for b in range(batch_size):
                fused_coeff += weight[b] * coeff[b, :, :]
        elif fusion_type == 'min':
            fused_coeff = np.min(coeff, axis=-1)
        elif fusion_type == 'max':
            fused_coeff = np.max(coeff, axis=-1)
        else:
            raise ValueError('error fusion type')

        return fused_coeff

    @staticmethod
    def _visualize_fusion(fused_coeff, fused_image, level):
        """
        可视化融合结果
        :param fused_coeff: 融合后的系数
        :param fused_image: 融合后的图像
        :param level: 小波分解的层级
        """
        plt.figure('DWT Fusion')
        plt.subplot(1, level + 2, 1)
        plt.imshow(fused_coeff[0], cmap='gray')
        plt.title('Fused cA')

        for l in range(1, level + 1):
            plt.subplot(1, level + 2, l + 1)
            plt.imshow(fused_coeff[l][0], cmap='gray')  # cH
            plt.title(f'Fused cH{l}')

        plt.subplot(1, level + 2, level + 2)
        plt.imshow(fused_image, cmap='gray')
        plt.title('Fused Image')
        plt.show(block=True)
