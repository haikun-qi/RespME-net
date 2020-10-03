
import os
from params import params
import numpy as np
from volume import DataGenerator
from train_generator import my_fit_generator
from datetime import datetime
import random
import time
import scipy.io as sio


if __name__ == '__main__':

    gpu_list_param = params['gpu_list']
    if isinstance(gpu_list_param, int):
        gpu_list = [gpu_list_param]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list_param)
    else:
        gpu_list = list(range(len(gpu_list_param.split(','))))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_param

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


    all_data_list=os.listdir(params['data_dir'])

    train_data_list = all_data_list

    ckpt_dir = params['ckpt_dir']+'Train'+datetime.now().strftime("%d%b_%I%M%P")+'/'

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)


    IDs = []

    num_sam_per_bin = 200
    for fname in train_data_list:
        fname = params['data_dir'] + fname
        for bin_num in range(1,4):
            rand_xyz = [[]]
            count_num=0
            while(count_num < num_sam_per_bin):
                xx = np.random.randint(0, 192 - 64 + 1, dtype='int32')
                yy = np.random.randint(0, 192 - 64 + 1, dtype='int32')
                zz = np.random.randint(0, 80 - 64 + 1, dtype='int32')
                if ([xx,yy,zz] not in rand_xyz):
                    ID={'name':fname,'bin_num': bin_num, 'xx':xx, 'yy':yy, 'zz':zz}
                    IDs = IDs + [ID]
                    rand_xyz += [[xx,yy,zz]]
                    count_num += 1


    indexes = np.arange(len(IDs))
    np.random.shuffle(indexes)

    train_IDs = [IDs[k] for k in indexes]
    train_generator = DataGenerator(train_IDs, batch_size=params['batch_size'], dim=(64, 64, 64), n_channels = 2, shuffle = True)

    lr = params['learning_rate']
    max_epoch = 10
    decay_step = 2

    start_time = time.time()
    my_fit_generator(params,
                     train_generator,
                     ckpt_dir,
                     learning_rate=lr,
                     lr_decay_step = decay_step,
                     lr_decay_rate=2.0,
                     epochs = max_epoch,
                     max_queue_size=30,
                     workers=8,
                     use_multiprocessing=True,
                     shuffle=True,
                    initial_epoch=0)
    end_time = time.time()
    print("Motion estimation time: '{}' hours".format((end_time - start_time)/3600.0))






