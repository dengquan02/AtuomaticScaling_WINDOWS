import tensorflow as tf
import numpy as np
from dias.model.FPN_Model import Dias_FPN
from dias.model.Unet_Model import Dias_Unet
from dias.dataIO.data import IonoDataManager
from dias.dataIO.dataPostProcess import get_minH_maxF
import segmentation_models as sm
import matplotlib.pyplot as plt
import os

ori_height = 520
ori_width = 520

def plot(y, name, idx):
    plt.subplot(3,3,1+3*idx)
    plt.imshow(y[0,:ori_height,:ori_width,0])
    plt.gca().invert_yaxis()
    plt.title(name + ':E')

    plt.subplot(3,3,2+3*idx)
    plt.imshow(y[0,:ori_height,:ori_width,1])
    plt.gca().invert_yaxis()
    plt.title(name + ':F1')
    
    plt.subplot(3,3,3+3*idx)
    plt.imshow(y[0,:ori_height,:ori_width,2])
    plt.gca().invert_yaxis()
    plt.title(name + ':F2')

def test(cfgs):
    print('Setting Model...')
    # set up model
    # if cfgs['Model']['Type'] == 'Unet' or cfgs['Model']['Type'] == 'naiveUnet':
    #     model = Dias_Unet(cfgs)
    # elif cfgs['Model']['Type'] == 'FPN':
    #     model = Dias_FPN(cfgs)
    # model.load_weights(cfgs['Test']['ModelPath'])
    model = tf.keras.models.load_model(cfgs['Test']['ModelPath'], compile=False) 

    threshold = float(cfgs['Test']['Threshold'])
    # set up dataset
    dataManager = IonoDataManager(cfgs)
    all_test_num = len(dataManager.test_data_list)

    # plot image save directory
    img_save_dir = cfgs['Test']['ImgSaveDir']
    if os.path.exists(img_save_dir):
        pass
    else:
        os.makedirs(img_save_dir)

    # if only save MinH and MaxF
    if cfgs['Test']['TestSave'] == 'OnlyMinHMaxF':
        res_mat = np.zeros([len(dataManager.test_data_list),3,6])
        for idx in range(len(dataManager.test_data_list)):
            test_data, human_res, artist_res = dataManager.get_test_batch(idx)
            Dias_res = model.predict(test_data)
            res_mat[idx,:,0:2] = get_minH_maxF(human_res,threshold)
            res_mat[idx,:,2:4] = get_minH_maxF(artist_res,threshold)
            res_mat[idx,:,4:6] = get_minH_maxF(Dias_res,threshold)

            plt.figure(figsize=(8,16))
            plt.subplot(1,4,1)
            plt.imshow(test_data[0,:ori_height,:ori_width,:])
            plt.gca().invert_yaxis()
            plt.title('x')
            plt.subplot(1,4,2)
            plt.imshow(test_data[0,:ori_height,:ori_width,0])
            plt.gca().invert_yaxis()
            plt.title('x' + ':O-mode recordings')
            plt.subplot(1,4,3)
            plt.imshow(test_data[0,:ori_height,:ori_width,1])
            plt.gca().invert_yaxis()
            plt.title('x' + ':X-mode recordings')
            plt.subplot(1,4,4)
            plt.imshow(test_data[0,:ori_height,:ori_width,2])
            plt.gca().invert_yaxis()
            plt.title('x' + ':most probably amplitude of O-mode at each frequency')

            plt.figure(figsize=(8,16))
            plt.subplot(1,4,1)
            plt.imshow(Dias_res[0,:ori_height,:ori_width,:])
            plt.gca().invert_yaxis()
            plt.title('y Dias_res')
            plt.subplot(1,4,2)
            plt.imshow(Dias_res[0,:ori_height,:ori_width,0])
            plt.gca().invert_yaxis()
            plt.title('y' + ':E Dias_res')
            plt.subplot(1,4,3)
            plt.imshow(Dias_res[0,:ori_height,:ori_width,1])
            plt.gca().invert_yaxis()
            plt.title('y' + ':F1 Dias_res')
            plt.subplot(1,4,4)
            plt.imshow(Dias_res[0,:ori_height,:ori_width,2])
            plt.gca().invert_yaxis()
            plt.title('y' + ':F2 Dias_res')

            plt.figure(figsize=(8,16))
            plt.subplot(1,4,1)
            plt.imshow(human_res[0,:ori_height,:ori_width,:])
            plt.gca().invert_yaxis()
            plt.title('y')
            plt.subplot(1,4,2)
            plt.imshow(human_res[0,:ori_height,:ori_width,0])
            plt.gca().invert_yaxis()
            plt.title('y' + ':E')
            plt.subplot(1,4,3)
            plt.imshow(human_res[0,:ori_height,:ori_width,1])
            plt.gca().invert_yaxis()
            plt.title('y' + ':F1')
            plt.subplot(1,4,4)
            plt.imshow(human_res[0,:ori_height,:ori_width,2])
            plt.gca().invert_yaxis()
            plt.title('y' + ':F2')


            plt.figure(figsize=(16,16))
            plot(human_res, 'human_res', 0)
            plot(artist_res, 'artist_res', 1)
            plot(Dias_res, 'Dias_res', 2)
            # plt.close()

            plt.figure(figsize=(18,6))
            plt.subplot(1,4,1)
            plt.imshow(test_data[0,:ori_height,:ori_width,0])
            plt.gca().invert_yaxis()
            plt.title('test_data') 
            plt.ylabel('VHSP(N)')
            plt.xlabel('CFSP(N)')
            plt.subplot(1,4,2)
            plt.imshow(human_res[0,:ori_height,:ori_width,0])
            plt.gca().invert_yaxis()
            plt.title('human_res') 
            plt.ylabel('VHSP(N)')
            plt.xlabel('CFSP(N)')
            plt.subplot(1,4,3)
            plt.imshow(artist_res[0,:ori_height,:ori_width,0])
            plt.gca().invert_yaxis()
            plt.title('artist_res')
            plt.ylabel('VHSP(N)')
            plt.xlabel('CFSP(N)')
            plt.subplot(1,4,4)
            plt.imshow(Dias_res[0,:ori_height,:ori_width,0])
            plt.gca().invert_yaxis()
            plt.title('Dias_res') 
            plt.ylabel('VHSP(N)')
            plt.xlabel('CFSP(N)')
            plt.savefig(img_save_dir+'IDX_{}.png'.format(idx),dpi=300)
            # plt.close()

            if idx%100==0:
                print(res_mat[idx,:])

        np.save(cfgs['Test']['SavePath']+'MinHMaxF.npy',res_mat)
    # if save All outputs of the model
    elif cfgs['Test']['TestSave'] == 'AllOutput':
        for idx in range(all_test_num):
            print('On {}'.format(idx))
            if cfgs['Test']['ScaleOnly']:
                test_data = dataManager.get_scale_only(idx)
                Dias_res = model.predict(test_data)
                res_mat = (test_data,Dias_res)
                np.save(cfgs['Test']['SavePath']+'TEST_{}.npy'.format(idx),res_mat)
            else:
                test_data, human_res, artist_res = dataManager.get_test_batch(idx)
                Dias_res = model.predict(test_data)
                res_mat = (test_data, human_res, artist_res,Dias_res)
                np.save(cfgs['Test']['SavePath']+'TEST_{}.npy'.format(idx),res_mat)
    else:
        print('Please choose a vaild TestSave Option')
    
    return
