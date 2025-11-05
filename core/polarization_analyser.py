import os
import re
import cv2
import glob
import numpy as np
from natsort import natsorted

__all__ = [
    'PDM',
    'PA'
]


eps = np.finfo(np.float64).eps


def cv2_imread(file_path, flags=cv2.IMREAD_UNCHANGED):
    with open(file_path, 'rb') as f:
        data = f.read()

    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, flags)

    return img


class PDM:
    """Polarization Dataset Manager"""
    DTYPE_DOT_FP = 0
    DTYPE_DOT_LP = 1
    DTYPE_DOF_LP = 2

    def __init__(self, dataset_type: int):
        self.dataset_type = dataset_type
        self.polarization_images = None  # [H, W, N]
        self.lp_angles = None
        self.qwp_angles = None

    def import_dof_polarization_image(self, imfile):
        file_extension = os.path.splitext(imfile)[1]
        assert file_extension.lower() in ('.jpg', '.bmp', '.png')

        if os.path.exists(imfile):
            dof_polarization_image = cv2_imread(imfile)
            if dof_polarization_image.ndim == 3:
                dof_polarization_image = cv2.cvtColor(dof_polarization_image, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError('image path does not exist')

        self.polarization_images, self.lp_angles = PDM.demosaicing(dof_polarization_image)

    @staticmethod
    def demosaicing(dof_polarization_image: np.ndarray):
        height = dof_polarization_image.shape[0]
        width = dof_polarization_image.shape[1]
        polarization_images = np.zeros((height, width, 4), dtype=np.float64)
        lp_angles = np.array([0, 45, 90, 135]) * np.pi / 180

        code_bg = getattr(cv2, f"COLOR_BayerBG2BGR")
        code_gr = getattr(cv2, f"COLOR_BayerGR2BGR")
        img_debayer_bg = cv2.cvtColor(dof_polarization_image, code_bg)
        img_debayer_gr = cv2.cvtColor(dof_polarization_image, code_gr)
        img_000, _, img_090 = cv2.split(img_debayer_bg)
        img_045, _, img_135 = cv2.split(img_debayer_gr)

        polarization_images[:, :, 0] = img_000.astype(float) / 255
        polarization_images[:, :, 1] = img_045.astype(float) / 255
        polarization_images[:, :, 2] = img_090.astype(float) / 255
        polarization_images[:, :, 3] = img_135.astype(float) / 255

        return polarization_images, lp_angles

    def import_dot_full_polarization_images(self, imdir):
        print('import full polarization images...')
        extensions = ['*.bmp', '*.png']
        polar_images_list = natsorted([file for ext in extensions for file in glob.glob(os.path.join(imdir, ext))])

        if not polar_images_list:
            raise ValueError('图像数据路径为空，请检查图像路径或图像格式！')

        # 导入偏振图像数据集
        lp_angles = []
        qwp_angles = []
        polarization_images = []
        for k in range(len(polar_images_list)):
            im = cv2_imread(polar_images_list[k])
            gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if len(im.shape) == 3 else im

            # 尝试根据图像名称得到波片方位角度
            img_name = os.path.basename(polar_images_list[k])
            name_without_ext = os.path.splitext(img_name)[0]
            try:
                numbers = re.findall(r'\d+', name_without_ext)
                numbers = [int(num) for num in numbers]
                lp_angles.append(np.deg2rad(numbers[0]))
                qwp_angles.append(np.deg2rad(numbers[1]))
                polarization_images.append(gray_im.astype(float) / 255)
            except IndexError:
                print(f"Warning: Skipping '{os.path.basename(polar_images_list[k])}'")

        print('数据集中共有 ', len(polar_images_list), ' 张图像')
        self.lp_angles = np.array(lp_angles)
        self.qwp_angles = np.array(qwp_angles)
        self.polarization_images = np.stack(polarization_images, axis=-1)

    def import_dot_linear_polarization_images(self, imdir):
        print('import linear polarization images...')
        extensions = ['*.bmp', '*.png']
        polar_images_list = natsorted([file for ext in extensions for file in glob.glob(os.path.join(imdir, ext))])

        if not polar_images_list:
            raise ValueError('图像数据路径为空，请检查图像路径或图像格式！')

        # 导入偏振图像数据集
        lp_angles = []
        polarization_images = []
        for k in range(len(polar_images_list)):
            img = cv2_imread(polar_images_list[k])
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            polarization_images.append(img.astype(float) / 255.)

            # 根据图像名称得到偏振方位角度
            image_name = os.path.basename(polar_images_list[k])
            number_str = ''.join(filter(str.isdigit, image_name))
            lp_angles.append(int(number_str) * np.pi / 180)

        print('数据集中共有 ', len(polar_images_list), ' 张图像')
        self.lp_angles = np.array(lp_angles)
        self.polarization_images = np.array(polarization_images).transpose(1, 2, 0)



class PA:
    """Polarization Analyser"""
    def __init__(self, pdm: PDM):
        self.pdm = pdm

        self.stokes = None
        self.rho = None
        self.phi = None
        self.chi = None
        self.iun = None
        self.imin_images = None
        self.iavg_images = None
        self.imax_images = None
        self.imin_stokes = None
        self.imax_stokes = None
        
    def calc_stokes(self, ignore_invalid=True):
        images = self.pdm.polarization_images
        if self.pdm.dataset_type == PDM.DTYPE_DOT_FP:
            lp_angles = self.pdm.lp_angles
            qwp_angles = self.pdm.qwp_angles
            stokes = self.calc_full_stokes(images, lp_angles, qwp_angles, ignore_invalid)
        elif self.pdm.dataset_type in [PDM.DTYPE_DOT_LP, PDM.DTYPE_DOF_LP]:
            lp_angles = self.pdm.lp_angles
            stokes = self.calc_linear_stokes(images, lp_angles, ignore_invalid)
        else:
            raise ValueError
        
        self.stokes = stokes

    @staticmethod
    def calc_full_stokes(images, lp_angles, qwp_angles, ignore_invalid=True, valid_range=[0, 0.99]):
        assert lp_angles.shape == qwp_angles.shape

        H, W, N = images.shape
        values = images.reshape(H * W, N)
        stokes = PA.solve_full_stokes_contradictory_equation(lp_angles, qwp_angles, values)

        s0 = stokes[:, 0]
        s1 = stokes[:, 1]
        s2 = stokes[:, 2]
        s3 = stokes[:, 3]

        if ignore_invalid is False:
            values_min = np.min(values, axis=1)
            values_max = np.max(values, axis=1)
            special_pixel_index = np.where((values_max >= valid_range[1]) | (values_min <= valid_range[0]))[0]
            print(f'{len(special_pixel_index)} special pixels found')

            for k in range(len(special_pixel_index)):
                values_pt = values[special_pixel_index[k], :]
                valid_index = np.where((values_pt <= valid_range[1]) & (values_pt >= valid_range[0]))[0]
                if 0 < len(valid_index) < 4:
                    valid_index = PA._expand_valid_index(valid_index)
                
                if len(valid_index) >= 4:
                    valid_values = values_pt[valid_index]
                    valid_lp_angles = lp_angles[valid_index]
                    valid_qwp_angles = qwp_angles[valid_index]
                    valid_stokes = PA.solve_full_stokes_contradictory_equation(
                        valid_lp_angles, valid_qwp_angles, valid_values)
                    
                    s0[special_pixel_index[k]] = valid_stokes[0]
                    s1[special_pixel_index[k]] = valid_stokes[1]
                    s2[special_pixel_index[k]] = valid_stokes[2]
                    s3[special_pixel_index[k]] = valid_stokes[3]

        stokes = stokes.reshape((H, W, 4))
        return stokes

    @staticmethod
    def solve_full_stokes_contradictory_equation(lp_angles, qwp_angles, values):
        """
        计算全斯托克斯矛盾方程组
        :param lp_angles: 线偏振片角度数组
        :param qwp_angles: 四分之一波片角度数组
        :param values: (HW, N)的二维数组
        :return: stokes: (4 * HW)的值数组
        """
        if values.ndim == 2:
            values = values.transpose()  # (N, HW)

        ones_column = np.ones((len(qwp_angles), 1))
        cos_cos_column = (np.cos(2 * qwp_angles) * np.cos(2 * lp_angles - 2 * qwp_angles)).reshape(-1, 1)
        sin_cos_column = (np.sin(2 * qwp_angles) * np.cos(2 * lp_angles - 2 * qwp_angles)).reshape(-1, 1)
        sin_column = (np.sin(2 * lp_angles - 2 * qwp_angles)).reshape(-1, 1)
        A = 0.5 * np.hstack((ones_column, cos_cos_column, sin_cos_column, sin_column))
        x = np.linalg.pinv(A, rcond=1e-15) @ values  # (4, HW)
        stokes = x.transpose()  # (HW, 4)
        return stokes

    @staticmethod
    def calc_linear_stokes(images:np.ndarray, lp_angles, ignore_invalid=True, valid_range=[0.0, 0.99]):
        H, W, N = images.shape
        values = images.reshape(H * W, N)
        stokes = PA.solve_linear_stokes_contradictory_equation(lp_angles, values)

        s0 = stokes[:, 0]
        s1 = stokes[:, 1]
        s2 = stokes[:, 2]

        if ignore_invalid is False:
            values_min = np.min(values, axis=1)
            values_max = np.max(values, axis=1)
            special_pixel_index = np.where((values_max >= valid_range[1]) | (values_min <= valid_range[0]))[0]
            print(f'{len(special_pixel_index)} special pixels found')

            for k in range(len(special_pixel_index)):
                values_pt = values[special_pixel_index[k], :]
                valid_index = np.where((values_pt <= valid_range[1]) & (values_pt >= valid_range[0]))[0]
                if len(values_pt) < 3:
                    valid_index = PA._expand_valid_index(valid_index)

                if len(valid_index) >= 3:
                    valid_values = values_pt[valid_index]
                    valid_lp_angles = lp_angles[valid_index]

                    valid_stokes = PA.solve_linear_stokes_contradictory_equation(
                        valid_lp_angles, valid_values)

                    s0[special_pixel_index[k]] = valid_stokes[0]
                    s1[special_pixel_index[k]] = valid_stokes[1]
                    s2[special_pixel_index[k]] = valid_stokes[2]

        stokes = stokes.reshape((H, W, 3))
        return stokes

    @staticmethod
    def solve_linear_stokes_contradictory_equation(lp_angles, values, ignore_valid=False):
        """
        求解线偏振矛盾方程组
        :param lp_angles: 线偏振片角度
        :param values: (HW * N) 的值数组 或 一维数组
        :return: stokes: (3 * HW) 的值数组
        """
        if values.ndim == 2:
            values = values.transpose()  # (N, HW)

        ones_column = np.ones((len(lp_angles), 1))
        cos_column = np.cos(2 * lp_angles).reshape(-1, 1)
        sin_column = np.sin(2 * lp_angles).reshape(-1, 1)
        A = 0.5 * np.hstack((ones_column, cos_column, sin_column))
        x = np.linalg.pinv(A, rcond=1e-15) @ values  # (4, HW)
        stokes = x.transpose()  # (HW, 4)
        return stokes

    @staticmethod
    def _expand_valid_index(valid_index):
        """扩充有效值，认为有效值边缘处的无效值为有效的，例如[0,1,2,6,7] -> [0,1,2,3,5,6,7]"""
        continuous_index = np.arange(valid_index.min(), valid_index.max() + 1)
        missing_index = np.setdiff1d(continuous_index, valid_index)

        if missing_index.size == 0:
            # missing_index 是一个空的 np 数组
            return valid_index

        min_index = np.array([missing_index.min()])
        max_index = np.array([missing_index.max()])
        expanded_valid_index = np.sort(np.concatenate([valid_index, min_index, max_index]))
        return expanded_valid_index

    @staticmethod
    def stokes2rho(stokes, s0_threshold=0.):
        ndims = stokes.ndim

        if ndims == 2:
            rho = np.sqrt(np.sum(stokes[:, 1:] ** 2, axis=-1)) / stokes[:, 0].clip(min=eps)
            rho[stokes[:, 0] <= s0_threshold] = 0
        elif ndims == 3:
            rho = np.sqrt(np.sum(stokes[:, :, 1:] ** 2, axis=-1)) / stokes[:, :, 0].clip(min=eps)
            rho[stokes[:, :, 0] <= s0_threshold] = 0
        else:
            raise ValueError(
                "Unsupported dimensions for stokes array. Expected 2 or 3 dimensions, got {}.".format(ndims))

        rho[rho > 1] = 1
        return rho

    @staticmethod
    def stokes2phi(stokes, s0_threshold=0.):
        ndims = stokes.ndim

        if ndims == 2:
            phi = 0.5 * np.arctan2(stokes[:, 2], stokes[:, 1])
            phi[stokes[:, 0] <= s0_threshold] = 0
        elif ndims == 3:
            phi = 0.5 * np.arctan2(stokes[:, :, 2], stokes[:, :, 1])
            phi[stokes[:, :, 0] <= s0_threshold] = 0
        else:
            raise ValueError(
                "Unsupported dimensions for stokes array. Expected 2 or 3 dimensions, got {}.".format(ndims))

        return phi

    @staticmethod
    def stokes2chi(stokes: np.ndarray) -> np.ndarray:
        # ellipticity angle \in [-pi/4, pi/4]
        chi = 0.5 * np.arctan2(stokes[:, :, 3], np.sqrt(stokes[:, :, 1] ** 2 + stokes[:, :, 2] ** 2))
        return chi

    @staticmethod
    def stokes2iun(stokes):
        ndims = stokes.ndim

        if ndims == 2:
            iun = stokes[:, 0] / 2
        elif ndims == 3:
            iun = stokes[:, :, 0] / 2
        else:
            raise ValueError(
                "Unsupported dimensions for stokes array. Expected 2 or 3 dimensions, got {}.".format(ndims))

        iun[iun > 1] = 1
        return iun

    @staticmethod
    def stokes2imin(stokes):
        imin = 0.5 * (stokes[:, :, 0] - np.sqrt(np.sum(stokes[:, :, 1:] ** 2, axis=-1)))
        imin[imin < 0] = 0
        return imin

    @staticmethod
    def stokes2imax(stokes):
        imax = 0.5 * (stokes[:, :, 0] + np.sqrt(np.sum(stokes[:, :, 1:] ** 2, axis=-1)))
        return imax

    @staticmethod
    def stokes2lp(stokes, lp_angle):
        s0 = stokes[:, :, 0]
        s1 = stokes[:, :, 1]
        s2 = stokes[:, :, 2]
        im_lp = 0.5 * (s0 + np.cos(2 * lp_angle) * s1 + np.sin(2 * lp_angle) * s2)
        return im_lp.clip(min=0, max=1)

    @staticmethod
    def images2iavg(images):
        iun_ = np.mean(images, axis=2)
        return iun_

    @staticmethod
    def images2imin(images):
        imin_ = np.min(images, axis=2)
        imin_[imin_ > 1] = 1
        return imin_

    @staticmethod
    def images2imax(images):
        imax_ = np.max(images, axis=2)
        return imax_

