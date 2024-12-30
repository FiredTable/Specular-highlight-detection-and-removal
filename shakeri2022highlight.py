import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import MyTools.polarization_analysis_tools as pat
from scipy.linalg import svd
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from MyTools.image_common import imnorm, ImageCropper, imsave
from MyTools.image_inpaint import yang2010real


def calc_polar_chromaticity_image(rgb_polar_analyser: pat.MultiChannelPolarizationAnalyser, visualize=False):
    # calc i_raw_d
    i_c = rgb_polar_analyser.mc_iun.copy().reshape(-1, 3)
    i_sv = (rgb_polar_analyser.mc_imax - rgb_polar_analyser.mc_imin).reshape(-1, 3) / 2
    i_raw_d = (i_c - 2 * i_sv).clip(min=1e-10)  # will be optimized by the further

    # calc i_min
    mean_i_min = np.mean(np.min(i_c, axis=-1))

    # calc i_chro
    i_chro_r = i_raw_d[:, 0] / (np.sum(i_raw_d, axis=-1) + mean_i_min)
    i_chro_g = i_raw_d[:, 1] / (np.sum(i_raw_d, axis=-1) + mean_i_min)
    i_chro_b = i_raw_d[:, 2] / (np.sum(i_raw_d, axis=-1) + mean_i_min)
    i_chro = np.stack((i_chro_r, i_chro_g, i_chro_b), axis=1)

    height, width = rgb_polar_analyser.mc_iun.shape[0: 2]
    i_c = i_c.reshape((height, width, 3))
    i_sv = i_sv.reshape((height, width, 3))
    i_raw_d = i_raw_d.reshape((height, width, 3))
    i_chro = i_chro.reshape((height, width, 3))

    if visualize:
        plt.figure()
        plt.subplot(221)
        plt.imshow(i_c)
        plt.subplot(222)
        plt.imshow(i_sv)
        plt.title('i_sv')
        plt.subplot(223)
        plt.imshow(i_raw_d)
        plt.subplot(224)
        plt.imshow(i_chro)
        plt.title('i_chro')
        plt.show(block=True)

    return i_chro


def create_multi_representations(i_chro, rgb_polar_images, num_candidate=4, similarity_thresh=0.03, block_size=3):
    """
    :param i_chro: polarization-based chromaticity image
    :param rgb_polar_images: rgb polarization images [H,W,B,C], [0, 45, 90, 135]
    :param num_candidate: Each of those n4 selected pixels can be considered as one representation of pixel p
    :param similarity_thresh: ||I_chro(p) − I_chro(q)|| < T
    :param block_size: the size of block, type=int, default=3
    :return: representations
    """
    height, width, batch, channels = rgb_polar_images.shape

    i0 = rgb_polar_images[:, :, 0, :]
    i45 = rgb_polar_images[:, :, 1, :]
    i90 = rgb_polar_images[:, :, 2, :]
    i135 = rgb_polar_images[:, :, 3, :]

    i0_representations = []
    i45_representations = []
    i90_representations = []
    i135_representations = []

    print(f'now create multiple representations...')
    block_arm_length = int((block_size - 1) / 2)
    for y in tqdm(range(height)):
        for x in range(width):
            block_chro = i_chro[max(0, y - block_arm_length): min(height, y + block_arm_length + 1),
                           max(0, x - block_arm_length): min(width, x + block_arm_length + 1), :]
            block_i0 = i0[max(0, y - block_arm_length): min(height, y + block_arm_length + 1),
                          max(0, x - block_arm_length): min(width, x + block_arm_length + 1), :]
            block_i45 = i45[max(0, y - block_arm_length): min(height, y + block_arm_length + 1),
                            max(0, x - block_arm_length): min(width, x + block_arm_length + 1), :]
            block_i90 = i90[max(0, y - block_arm_length): min(height, y + block_arm_length + 1),
                            max(0, x - block_arm_length): min(width, x + block_arm_length + 1), :]
            block_i135 = i135[max(0, y - block_arm_length): min(height, y + block_arm_length + 1),
                              max(0, x - block_arm_length): min(width, x + block_arm_length + 1), :]

            flat_block_chro = block_chro.reshape(-1, channels)
            flat_block_i0 = block_i0.reshape(-1, channels)
            flat_block_i45 = block_i45.reshape(-1, channels)
            flat_block_i90 = block_i90.reshape(-1, channels)
            flat_block_i135 = block_i135.reshape(-1, channels)

            chromaticity = i_chro[y, x, :]

            candidates_i0 = []
            candidates_i45 = []
            candidates_i90 = []
            candidates_i135 = []
            for k in range(len(flat_block_chro)):
                q_chro = flat_block_chro[k, :]
                q_i0 = flat_block_i0[k, :]
                q_i45 = flat_block_i45[k, :]
                q_i90 = flat_block_i90[k, :]
                q_i135 = flat_block_i135[k, :]

                dist = np.linalg.norm(chromaticity - q_chro)

                if dist < similarity_thresh:
                    candidates_i0.append(q_i0)
                    candidates_i45.append(q_i45)
                    candidates_i90.append(q_i90)
                    candidates_i135.append(q_i135)

            if len(candidates_i0) >= num_candidate:
                selected_idx = np.random.choice(len(candidates_i0), num_candidate, replace=False)
                selected_candidates_i0 = [candidates_i0[idx] for idx in selected_idx]
                selected_candidates_i45 = [candidates_i45[idx] for idx in selected_idx]
                selected_candidates_i90 = [candidates_i90[idx] for idx in selected_idx]
                selected_candidates_i135 = [candidates_i135[idx] for idx in selected_idx]
            else:
                # repeatedly choose
                selected_idx = np.random.choice(len(candidates_i0), num_candidate, replace=True)
                selected_candidates_i0 = [candidates_i0[idx] for idx in selected_idx]
                selected_candidates_i45 = [candidates_i45[idx] for idx in selected_idx]
                selected_candidates_i90 = [candidates_i90[idx] for idx in selected_idx]
                selected_candidates_i135 = [candidates_i135[idx] for idx in selected_idx]

            i0_representations.append(selected_candidates_i0)
            i45_representations.append(selected_candidates_i45)
            i90_representations.append(selected_candidates_i90)
            i135_representations.append(selected_candidates_i135)

    # 将候选表示转换为形状为 (height, width, num_candidate, channels) 的数组
    np_representations = [np.array(i0_representations).reshape((height, width, num_candidate, channels)),
                          np.array(i45_representations).reshape((height, width, num_candidate, channels)),
                          np.array(i90_representations).reshape((height, width, num_candidate, channels)),
                          np.array(i135_representations).reshape((height, width, num_candidate, channels))]
    return np_representations


def initialize_diffuse_components(representations):
    height, width, num_representation, channels = representations.shape
    initial_diffuse = np.zeros((height, width, num_representation, channels))

    print('now initialize diffuse components...')
    for k in tqdm(range(num_representation)):
        # chromatic-based hr method by yang2010real
        initial_diffuse[:, :, k, :] = yang2010real(representations[:, :, k, :])

    return initial_diffuse


def tensor_unfold(tensor, mode=2):
    """[a, b, c] -> [ab, c]"""
    return np.reshape(tensor, (np.prod(tensor.shape[:mode]), -1))


def tensor_fold(tensor, shape):
    return np.reshape(tensor, shape)


def L2phi2Q(L, height, channels, num_representations):
    l_split = np.split(L, 4, axis=0)
    l_i0 = l_split[0].reshape(height, -1, channels, num_representations)
    l_i45 = l_split[1].reshape(height, -1, channels, num_representations)
    l_i90 = l_split[2].reshape(height, -1, channels, num_representations)
    l_i135 = l_split[2].reshape(height, -1, channels, num_representations)

    l_i0_new = []
    l_i45_new = []
    l_i90_new = []
    l_i135_new = []

    for k in range(num_representations):
        i0 = l_i0[:, :, :, k]
        i45 = l_i45[:, :, :, k]
        i90 = l_i90[:, :, :, k]
        i135 = l_i135[:, :, :, k]

        rgb_polarization_images = np.stack((i0, i45, i90, i135), axis=2)  # [H,W,B,C]
        lp_angles = np.array([0., np.pi / 4, np.pi / 2, np.pi / 4 * 3])

        rgb_polar_analyser = pat.MultiChannelPolarizationAnalyser(pat.DTYPE_DOTLINEARPOL)
        rgb_polar_analyser.import_multi_channel_linear_polarization_data(rgb_polarization_images, lp_angles)

        rgb_polar_analyser.create_multi_channel_polar_analyser()
        rgb_polar_analyser.calc_polar_features(verbose=False, visualize=False)
        phi_mean = np.mean(rgb_polar_analyser.mc_phi, axis=-1)

        ic = rgb_polar_analyser.mc_iun.reshape(-1, 3)
        isv = (rgb_polar_analyser.mc_imax - rgb_polar_analyser.mc_imin).reshape(-1, 3) / 2
        i0_new_r = ic[:, 0] + isv[:, 0] * np.cos(2 * 0 - 2 * phi_mean.reshape(-1,))
        i0_new_g = ic[:, 1] + isv[:, 1] * np.cos(2 * 0 - 2 * phi_mean.reshape(-1,))
        i0_new_b = ic[:, 2] + isv[:, 2] * np.cos(2 * 0 - 2 * phi_mean.reshape(-1,))
        i45_new_r = ic[:, 0] + isv[:, 0] * np.cos(2 * 45 - 2 * phi_mean.reshape(-1,))
        i45_new_g = ic[:, 1] + isv[:, 1] * np.cos(2 * 45 - 2 * phi_mean.reshape(-1,))
        i45_new_b = ic[:, 2] + isv[:, 2] * np.cos(2 * 45 - 2 * phi_mean.reshape(-1,))
        i90_new_r = ic[:, 0] + isv[:, 0] * np.cos(2 * 90 - 2 * phi_mean.reshape(-1,))
        i90_new_g = ic[:, 1] + isv[:, 1] * np.cos(2 * 90 - 2 * phi_mean.reshape(-1,))
        i90_new_b = ic[:, 2] + isv[:, 2] * np.cos(2 * 90 - 2 * phi_mean.reshape(-1,))
        i135_new_r = ic[:, 0] + isv[:, 0] * np.cos(2 * 135 - 2 * phi_mean.reshape(-1,))
        i135_new_g = ic[:, 1] + isv[:, 1] * np.cos(2 * 135 - 2 * phi_mean.reshape(-1,))
        i135_new_b = ic[:, 2] + isv[:, 2] * np.cos(2 * 135 - 2 * phi_mean.reshape(-1,))

        i0_new = np.stack((i0_new_r, i0_new_g, i0_new_b), axis=1).reshape(height, -1, channels)
        i45_new = np.stack((i45_new_r, i45_new_g, i45_new_b), axis=1).reshape(height, -1, channels)
        i90_new = np.stack((i90_new_r, i90_new_g, i90_new_b), axis=1).reshape(height, -1, channels)
        i135_new = np.stack((i135_new_r, i135_new_g, i135_new_b), axis=1).reshape(height, -1, channels)

        l_i0_new.append(i0_new)
        l_i45_new.append(i45_new)
        l_i90_new.append(i90_new)
        l_i135_new.append(i135_new)

    l_i0_new = np.array(l_i0_new).transpose(1, 2, 3, 0)
    l_i45_new = np.array(l_i45_new).transpose(1, 2, 3, 0)
    l_i90_new = np.array(l_i90_new).transpose(1, 2, 3, 0)
    l_i135_new = np.array(l_i135_new).transpose(1, 2, 3, 0)

    L_new = np.vstack((l_i0_new, l_i45_new, l_i90_new, l_i135_new)).reshape(4 * height, -1, num_representations)

    return L_new


def calc_tau(D, alpha=2, beta=0.25):
    i = D[:, :, 0]
    grad_i = np.gradient(i)[0] ** 2 + np.gradient(i)[1] ** 2
    grad_i = np.sqrt(grad_i)
    grad_I_smoothed = gaussian_filter(grad_i, sigma=1)
    tau = (1 - i) / np.exp(-alpha * np.power(grad_I_smoothed, beta))
    tau = np.stack((tau,) * 4, axis=-1)
    return tau


def tensor_low_rank_and_sparse_decomposition(D, lam, gamma, max_iter=50, tol=1e-3):
    """
    tensor low-rank and sparse decomposition
    :param D: input tensor [height*4, width*channels, num_representations]
    :param lam: the weight of sparse term
    :param gamma: the weight of normalized phase angle
    :param max_iter: max iterations
    :param tol: tolerance
    :return: L, S
    """
    height_4, width_channels, num_representations = D.shape
    height = height_4 // 4
    width = width_channels // 3
    channels = 3

    L = np.zeros_like(D)
    S = np.zeros_like(D)
    Y = np.zeros_like(D)

    mu = 1e-3
    rho = 1.1
    mu_max = 1e10

    print(f' now is tensor_low_rank_and_sparse_decomposition...')
    for iteration in tqdm(range(max_iter)):
        # update L
        X = D - S + (1 / mu) * Y
        X_unfolded = tensor_unfold(X, mode=2)

        # svd composition
        U, s, Vh = svd(X_unfolded, full_matrices=False)

        # singular value process
        threshold = 1 / (mu + 1)
        s = np.maximum(s - threshold, 0)

        # reconstruction low rank approximation
        L_unfolded = U @ np.diag(s) @ Vh
        L = tensor_fold(L_unfolded, D.shape)

        # phase angle regularization
        L_new = L2phi2Q(L, height, channels, num_representations)
        phase_angle_reg = gamma * np.linalg.norm(tensor_unfold(L) - tensor_unfold(L_new), 'fro') ** 2  # 计算相位角正则化项

        # update S
        tau = calc_tau(D)
        S = np.sign(D - L + (1/mu) * Y) * np.maximum(np.abs(D - L + (1/mu) * Y) - lam * tau / mu, 0)

        # update Y
        Y += mu * (D - L - S + phase_angle_reg)

        # update mu
        mu = min(rho * mu, mu_max)

        # calc error
        error = (np.linalg.norm(tensor_unfold(D) - tensor_unfold(L) - tensor_unfold(S), 'fro')
                 / np.linalg.norm(tensor_unfold(D), 'fro'))
        if error < tol:
            break

    return L, S


def average_L(L, channels, height):
    l = np.mean(L, axis=-1)
    l_split = np.split(l, 4, axis=0)
    l_i0 = l_split[0].reshape(height, -1, channels)
    l_i45 = l_split[1].reshape(height, -1, channels)
    l_i90 = l_split[2].reshape(height, -1, channels)
    l_i135 = l_split[2].reshape(height, -1, channels)

    return (l_i0 + l_i45 + l_i90 + l_i135) / 4


def single_image_polar_hr_dataset():
    datadir = r'E:\NPU\Research\polarization_image_data'
    subdir = r'PolarHR_dataset\scene1'

    name_list = os.listdir(os.path.join(datadir, subdir))

    groups = []
    for name in name_list:
        group_number = name.split('-')[0]

        # 如果组号还没有在字典中，创建一个新的列表
        if group_number not in groups:
            groups.append(group_number)

    group_name = '1'

    save_path = os.path.join('..', 'results', subdir, group_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    i0 = imnorm(skimage.io.imread(os.path.join(datadir, subdir, group_name + '-0.bmp')))
    i45 = imnorm(skimage.io.imread(os.path.join(datadir, subdir, group_name + '-45.bmp')))
    i90 = imnorm(skimage.io.imread(os.path.join(datadir, subdir, group_name + '-90.bmp')))
    i135 = imnorm(skimage.io.imread(os.path.join(datadir, subdir, group_name + '-135.bmp')))

    rgb_polarization_images = np.stack((i0, i45, i90, i135), axis=2)  # [H,W,B,C]
    lp_angles = np.array([0., np.pi / 4, np.pi / 2, np.pi / 4 * 3])

    rgb_polar_analyser = pat.MultiChannelPolarizationAnalyser(pat.DTYPE_DOTLINEARPOL, resolution_reduction=True)
    rgb_polar_analyser.import_multi_channel_linear_polarization_data(rgb_polarization_images, lp_angles)

    crop_coordinates = [None, None, None, None]
    if any(coord is None for coord in crop_coordinates):
        image_cropper = ImageCropper(rgb_polarization_images[:, :, 0, 0], crop_coordinates)
        crop_coordinates = image_cropper.crop_coordinates
    rgb_polar_analyser.crop_multi_channel_polarization_images(crop_coordinates)

    rgb_polar_analyser.create_multi_channel_polar_analyser()
    rgb_polar_analyser.calc_stokes(verbose=True, visualize=True)
    rgb_polar_analyser.calc_polar_features(verbose=True, visualize=True)

    i_chro = calc_polar_chromaticity_image(rgb_polar_analyser, visualize=True)
    representations = create_multi_representations(i_chro, rgb_polar_analyser.mc_polarization_images)
    i0_initial_diffuse = initialize_diffuse_components(representations[0]).transpose(0, 1, 3, 2)  # [H,W,B,C]->[H,W,C,B]
    i45_initial_diffuse = initialize_diffuse_components(representations[1]).transpose(0, 1, 3, 2)  # [H,W,B,C]->[H,W,C,B]
    i90_initial_diffuse = initialize_diffuse_components(representations[2]).transpose(0, 1, 3, 2)  # [H,W,B,C]->[H,W,C,B]
    i135_initial_diffuse = initialize_diffuse_components(representations[3]).transpose(0, 1, 3, 2)  # [H,W,B,C]->[H,W,C,B]

    height, width, channels, num_representations = i0_initial_diffuse.shape
    i0_initial_diffuse = i0_initial_diffuse.reshape(height, -1, num_representations)
    i45_initial_diffuse = i45_initial_diffuse.reshape(height, -1, num_representations)
    i90_initial_diffuse = i90_initial_diffuse.reshape(height, -1, num_representations)
    i135_initial_diffuse = i135_initial_diffuse.reshape(height, -1, num_representations)

    i_initial_diffuse = np.vstack(
        (i0_initial_diffuse, i45_initial_diffuse, i90_initial_diffuse, i135_initial_diffuse))

    D = i_initial_diffuse
    lam = 1 / np.sqrt(max(height * width, channels) * num_representations)
    gamma = 1e-6

    # 进行张量低秩和稀疏分解
    L, S = tensor_low_rank_and_sparse_decomposition(D, lam, gamma)
    i_diffuse = average_L(L, channels, height)
    i_specular = average_L(S, channels, height)
    imsave(i_diffuse, 'shakeri_diffuse.png', save_path, norm=True)
    imsave(i_specular, 'shakeri_specular.png', save_path)


def main():
    single_image_polar_hr_dataset()


if __name__ == '__main__':
    main()
