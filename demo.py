import os
import time
import argparse
from pathlib import Path
from core.common import imsave, Logger
from core.polarization_analyser import PDM, PA
from core.polarization_shdr import PolarFusionSHDR


def demo(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'logger.log')
    logger = Logger(log_path).init_logger()
    logger.info('\n-------------- Start --------------')

    pdm = PDM(args.dataset_type)
    if pdm.dataset_type == PDM.DTYPE_DOT_FP:
        pdm.import_dot_full_polarization_images(imdir=args.input)

    # step1: calculate stokes
    st_time = time.time()
    pa = PA(pdm)
    pa.calc_stokes(ignore_invalid=args.ignore_invalid)
    time_stokes = time.time() - st_time
    logger.info(f'Stokes calculation time: {time_stokes:.4f} s')
    logger.info(f'S0 range: min-({pa.stokes[:,:,0].min()}), max-({pa.stokes[:,:,0].max()})')
    logger.info(f'S1 range: min-({pa.stokes[:,:,1].min()}), max-({pa.stokes[:,:,1].max()})')
    logger.info(f'S2 range: min-({pa.stokes[:,:,2].min()}), max-({pa.stokes[:,:,2].max()})')
    imsave(pa.stokes[:,:,0], 's0.png', save_path=args.output_dir, norm=True, cmap_name='turbo')
    imsave(pa.stokes[:,:,1], 's1.png', save_path=args.output_dir, norm=True, cmap_name='turbo')
    imsave(pa.stokes[:,:,2], 's2.png', save_path=args.output_dir, norm=True, cmap_name='turbo')

    # step2: calculate polarization features
    st_time = time.time()
    pa.iun = pa.stokes2iun(pa.stokes)
    pa.rho = pa.stokes2rho(pa.stokes)
    pa.phi = pa.stokes2phi(pa.stokes)
    pa.imin_images = pa.images2imin(pdm.polarization_images)
    pa.iavg_images = pa.images2iavg(pdm.polarization_images)
    if pdm.dataset_type == PDM.DTYPE_DOT_FP:
        pa.chi = pa.stokes2chi(pa.stokes)
    time_polar = time.time() - st_time
    logger.info(f'Polarization features calculation time: {time_polar:.4f} s')
    imsave(pa.iun, 'iun.png', save_path=args.output_dir, norm=True, cmap_name='turbo')
    imsave(pa.rho, 'rho.png', save_path=args.output_dir, norm=True, cmap_name='turbo')
    imsave(pa.phi, 'phi.png', save_path=args.output_dir, norm=True, cmap_name='twilight_shifted')
    if pdm.dataset_type == PDM.DTYPE_DOT_FP:
        imsave(pa.chi, 'chi.png', save_path=args.output_dir, norm=True, cmap_name='twilight_shifted')
    
    # step3: detect specular highlight
    pfshdr = PolarFusionSHDR(pa)
    st_time = time.time()
    pfshdr.detect_specular_highlight(
        cluster_src='polar', canopy_t1=0.6, canopy_t2=0.4, mini_batch=False, r1=1.8, r2=0.8)
    time_shdr_detect = time.time() - st_time
    logger.info(f'Specular highlight detection time: {time_shdr_detect:.4f} s')
    for k, (iun_center, rho_center) in enumerate(zip(pfshdr.iun_centers, pfshdr.rho_centers)):
        logger.info(f'Cluster {k}: iun-({iun_center:.4f}), rho-({rho_center:.4f})')
    imsave(pfshdr.label_image, 'label_image.png', save_path=args.output_dir, norm=True, cmap_name='coolwarm')
    pfshdr.save_label_results(os.path.join(args.output_dir, 'label_image_with_ticks.png'))
    imsave(pfshdr.high_spec_mask, 'mask_high_spec.png', save_path=args.output_dir)
    imsave(pfshdr.spec2diffuse_mask, 'mask_spec2diffuse.png', save_path=args.output_dir)
    imsave(pfshdr.ext_spec_mask, 'mask_ext_spec.png', save_path=args.output_dir)
    mask_on_img = pfshdr.overlay2mask(pa.iavg_images)
    imsave(mask_on_img, 'mask on img.png', save_path=args.output_dir)
    imsave(pfshdr.outliers_mask1, 'mask1_outliers.png', save_path=args.output_dir)
    imsave(pfshdr.outliers_mask2, 'mask2_outliers.png', save_path=args.output_dir)

    # step4: remove specular highlight
    st_time = time.time()
    pfshdr.remove_specular_highlight(enhance=True)
    time_shdr_removal = time.time() - st_time
    logger.info(f'Specular highlight removal time: {time_shdr_removal:.4f} s')
    imsave(pa.imin_images, 'min_image.png', args.output_dir, norm=False)
    imsave(pa.iavg_images, 'avg_image.png', args.output_dir, norm=True)
    imsave(pfshdr.specular_source, 'init_fused_image.png', args.output_dir, norm=True)
    imsave(pfshdr.non_specular_source, 'ehe-avg_image.png', args.output_dir, norm=False)
    imsave(pfshdr.blending_image, 'blending_image.png', args.output_dir, norm=False)
    imsave(pfshdr.diffuse_image, 'diffuse_image.png', args.output_dir, norm=False)
    imsave(pfshdr.spec_image, 'spec_image.png', args.output_dir, norm=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=r".\input\scene1")
    parser.add_argument('--output_dir', default=r".\output\scene1")
    parser.add_argument('--ignore_invalid', default=True, type=bool)
    parser.add_argument('--dataset_type', default=PDM.DTYPE_DOT_FP)

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, f'ignore_invalid_{args.ignore_invalid}')
    demo(args)