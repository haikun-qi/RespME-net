
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from params import params

from volume import Test_Volume
import time
import scipy.io as sio

from scipy.ndimage import gaussian_filter

cwd = params['ckpt_dir']
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main(agrs=0):
    #%% choose a model from savedModels directory
    subDirectory='Train_st27_num9fv'

    #%% Load existing model.
    print ('Now loading the model ...')
    modelDir= cwd+subDirectory #complete path

    tf.reset_default_graph()
    loadChkPoint=tf.train.latest_checkpoint(modelDir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        new_saver = tf.train.import_meta_graph(modelDir+'/modelTst.meta')
        new_saver.restore(sess, loadChkPoint)
        graph = tf.get_default_graph()
        Iref=graph.get_tensor_by_name('Iref:0')
        Imov   =graph.get_tensor_by_name('Imov:0')
        flow_patch = graph.get_tensor_by_name('flowTst:0')
        # Iwarp = graph.get_tensor_by_name('Iwarp:0')

        data = Test_Volume(params, Iref, Imov)
        step_size = params['flow_stride'] #### sliding window step
        sz = params['mask_sz'] = 5 #### the size of border to be removed for each patch
        ref_bin = 0 ### reference bin
        mov_bins = [1, 2, 3] ### other 3 moving bins

        for vol_ind in range(0,data._num_volumes):
            data.load(vol_ind) ## update volume data
            kx, ky, kz, nbins = data.img.shape[0], data.img.shape[1], data.img.shape[2], data.img.shape[3]
            flow = np.zeros([kx, ky, kz, 3, nbins - 1], dtype=np.float32)
            mask = np.zeros([kx, ky, kz], dtype=np.float32)
            # img_warp = np.zeros([kx, ky, kz, nbins - 1], dtype=np.float32)

            xx_range = np.ndarray.tolist(np.array(range(0, kx - data.patch_size[0] + 1, step_size))) \
                       + [kx - data.patch_size[0]]
            yy_range = np.ndarray.tolist(np.array(range(0, ky - data.patch_size[1] + 1, step_size))) \
                       + [ky - data.patch_size[1]]
            zz_range = np.ndarray.tolist(np.array(range(0, kz - data.patch_size[2] + 1, step_size))) \
                       + [kz - data.patch_size[2]]


            st_time = time.time()
            for m in xx_range:
                for n in yy_range:
                    for k in zz_range:
                        eval_dict = data.get_eval_feed_dict([m, n, k], ref_bin=ref_bin, mov_bin=mov_bins)

                        rec = sess.run(flow_patch, feed_dict=eval_dict)
                        rec = np.squeeze(rec)
                        if np.ndim(rec) == 5:
                            rec = np.transpose(rec, [1, 2, 3, 4, 0])

                        flow[m + sz:m + data.patch_size[0] - sz, n + sz:n + data.patch_size[1] - sz,
                        k + sz:k + data.patch_size[2] - sz, ...] += \
                            rec[sz:data.patch_size[0] - sz, sz:data.patch_size[1] - sz, sz:data.patch_size[2] - sz,
                            ...]
                        mask[m + sz:m + data.patch_size[0] - sz, n + sz:n + data.patch_size[1] - sz,
                        k + sz:k + data.patch_size[2] - sz, ...] += 1

                        # rec = sess.run(Iwarp, feed_dict=eval_dict)
                        # rec = np.squeeze(rec)
                        # if np.ndim(rec) == 4:
                        #     rec = np.transpose(rec, [1, 2, 3, 0])
                        #     img_warp[m + sz:m + data.patch_size[0] - sz, n + sz:n + data.patch_size[1] - sz,
                        #     k + sz:k + data.patch_size[2] - sz, ...] += \
                        #         rec[sz:data.patch_size[0] - sz, sz:data.patch_size[1] - sz,
                        #         sz:data.patch_size[2] - sz,
                        #         ...]


            mask[mask == 0] = 1
            flow = flow / mask[..., np.newaxis, np.newaxis]
            eval_time = time.time() - st_time
            vol_name = os.path.split(data._volume_list[vol_ind])[-1]
            case_name = vol_name[0:-4]
            print('{}- Eval time: {}'.format(case_name, eval_time))


            for n in range(flow.shape[-1]):
                for m in range(flow.shape[-2]):
                    flow[:, :, :, m, n] = gaussian_filter(flow[:, :, :,m, n], sigma=params['sigma'])
            plot_flow(flow, show_slice=60)


            # img_warp = img_warp / mask[..., np.newaxis]
            # # show difference image
            # img_mov = np.delete(data.img, ref_bin, axis=-1)
            # diff_ori = np.abs(img_mov - data.img[..., ref_bin][..., np.newaxis])
            # diff_warp = np.abs(img_warp - data.img[..., ref_bin][..., np.newaxis])
            # plot_bins(diff_ori, diff_warp, ind_show=60)

            # save estimated motion
            # sio.savemat(params['res_dir'] + case_name  + '_netp.mat',
            #             {'flow_netp': flow})



def plot_flow(flow, show_slice=None):
    '''

   show 1 slice of the 3-bin motion
    '''
    flow=np.sqrt(np.squeeze(np.sum(np.square(flow),axis=-2)))

    if show_slice ==None:
        sli = flow.shape[2]//2
    else:
        sli = show_slice
    # %% Display the output images
    plot = lambda x: plt.imshow(x, cmap='jet',vmin=0.0,vmax=5.0)

    plt.clf()
    plt.subplot(131)
    plot(flow[:,:,sli,0])
    plt.axis('off')
    plt.title('Motion-Bin1')
    plt.subplot(132)
    plot(flow[:,:,sli, 1])
    plt.title('Motion-Bin2')
    plt.axis('off')
    plt.subplot(133)
    plot(flow[:,:,sli,2])
    plt.title('Motion-Bin3')
    plt.axis('off')
    plt.show()



def plot_bins(normAtb, normRec, ind_show=60):
    '''

       show 1 slice of the bin images
    '''

    normAtb = np.squeeze(normAtb[:,:,ind_show,:])
    normRec = np.squeeze(normRec[:,:,ind_show,:])
    nbins = normRec.shape[-1]
    #%% Display the output images
    plot = lambda x: plt.imshow(x,cmap='gray',clim=(0.0, .8))
    plt.clf()
    for nb in range(1,nbins+1):
        plt.subplot(2,nbins,nb)
        plot(normAtb[:,:,nb-1])
        plt.axis('off')
        plt.title('Original-diff')

        plt.subplot(2,nbins,nb+nbins)
        plot(normRec[:,:,nb-1])
        plt.title('Warped-diff')
        plt.axis('off')
    plt.show()

if __name__=='__main__':
    tf.app.run()