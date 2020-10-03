
import os
cwd = os.getcwd()

# build params
params = dict()
params['gpu_list'] = 1#'0,1' #0

params['data_dir'] = cwd + '/train/'
params['test_dir'] = cwd + '/test/'#''
params['res_dir'] = cwd  + '/motion_res/'
params['ckpt_dir'] = cwd + '/model/'


params['num_input_threads'] = 4
params['learning_rate'] = 1.0e-4
params['display_interval'] = 150


# Use one additional upconv layers to expand to full resolution in final network.
# If false, uses bilinear upsampling (x2).
params['full_res'] = False

# Total batch size, ust be divisible by the number of GPUs.
params['batch_size'] = 16
# params['batch_norm'] = True

# patch-based training
params['patch_size'] = [64, 64, 64]
# settings for sliding-window motion estimation
params['flow_stride'] = 48
params['mask_sz'] = 5
# gaussian filtering to smooth the final motion
params['sigma'] = 3

params['pyramid_loss'] = True

# Mask border regions in data term
params['border_mask'] = True



# -----------------------------------------------------------------------------
# Regularization

# Use second order smoothness
params['smooth_2nd_weight'] = 0.025
# other: gradient loss, NCC loss
params['gradient_weight'] = 0.5
params['ncc_weight'] = 1.0




