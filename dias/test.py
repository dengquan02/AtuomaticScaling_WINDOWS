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
    # set up model（根据配置加载已经训练好的模型）
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
        # res_mat 记录每个样本的 MinH / MaxF 指标：
        # 维度含义：[样本数, 3 条曲线(E/F1/F2), 6 个值(人工/专家/模型 的 MinH/MaxF)]
        res_mat = np.zeros([len(dataManager.test_data_list), 3, 6])
        for idx in range(len(dataManager.test_data_list)):
            # 取出当前样本的输入数据 test_data，以及人工标注 human_res、专家标注 artist_res
            test_data, human_res, artist_res = dataManager.get_test_batch(idx)
            # 模型预测结果 Dias_res，形状与 human_res 类似
            Dias_res = model.predict(test_data)
            # 计算人工标注的 MinH / MaxF
            res_mat[idx,:,0:2] = get_minH_maxF(human_res,threshold)
            # 计算专家标注的 MinH / MaxF
            res_mat[idx,:,2:4] = get_minH_maxF(artist_res,threshold)
            # 计算模型预测结果的 MinH / MaxF
            res_mat[idx,:,4:6] = get_minH_maxF(Dias_res,threshold)

            # # -------------------- 画图部分 1：输入数据各通道 --------------------
            # # 这一组图主要展示输入 ionogram 的各个通道，方便从原始数据角度理解样本
            # plt.figure(figsize=(8,16))
            # # 子图 1：三通道叠加显示的原始输入（RGB 形式）
            # plt.subplot(1,4,1)
            # plt.imshow(test_data[0,:ori_height,:ori_width,:])
            # plt.gca().invert_yaxis()
            # plt.title('x')
            # # 子图 2：通道 0，表示 O 模式记录
            # plt.subplot(1,4,2)
            # plt.imshow(test_data[0,:ori_height,:ori_width,0])
            # plt.gca().invert_yaxis()
            # plt.title('x' + ':O-mode recordings')
            # # 子图 3：通道 1，表示 X 模式记录
            # plt.subplot(1,4,3)
            # plt.imshow(test_data[0,:ori_height,:ori_width,1])
            # plt.gca().invert_yaxis()
            # plt.title('x' + ':X-mode recordings')
            # # 子图 4：通道 2，表示每个频率点最可能的 O 模式幅度
            # plt.subplot(1,4,4)
            # plt.imshow(test_data[0,:ori_height,:ori_width,2])
            # plt.gca().invert_yaxis()
            # plt.title('x' + ':most probably amplitude of O-mode at each frequency')

            # # -------------------- 画图部分 2：模型输出各通道 --------------------
            # # 这一组图展示模型预测的三通道结果，对应 E / F1 / F2 三条层
            # plt.figure(figsize=(8,16))
            # # 子图 1：三通道叠加显示的模型输出
            # plt.subplot(1,4,1)
            # plt.imshow(Dias_res[0,:ori_height,:ori_width,:])
            # plt.gca().invert_yaxis()
            # plt.title('y Dias_res')
            # # 子图 2：模型预测的 E 层
            # plt.subplot(1,4,2)
            # plt.imshow(Dias_res[0,:ori_height,:ori_width,0])
            # plt.gca().invert_yaxis()
            # plt.title('y' + ':E Dias_res')
            # # 子图 3：模型预测的 F1 层
            # plt.subplot(1,4,3)
            # plt.imshow(Dias_res[0,:ori_height,:ori_width,1])
            # plt.gca().invert_yaxis()
            # plt.title('y' + ':F1 Dias_res')
            # # 子图 4：模型预测的 F2 层
            # plt.subplot(1,4,4)
            # plt.imshow(Dias_res[0,:ori_height,:ori_width,2])
            # plt.gca().invert_yaxis()
            # plt.title('y' + ':F2 Dias_res')

            # # -------------------- 画图部分 3：人工标注各通道 --------------------
            # # 这一组图展示人工标注的 E / F1 / F2 三个通道，用来与模型输出对比
            # plt.figure(figsize=(8,16))
            # # 子图 1：三通道叠加显示的人标结果
            # plt.subplot(1,4,1)
            # plt.imshow(human_res[0,:ori_height,:ori_width,:])
            # plt.gca().invert_yaxis()
            # plt.title('y')
            # # 子图 2：人工标注的 E 层
            # plt.subplot(1,4,2)
            # plt.imshow(human_res[0,:ori_height,:ori_width,0])
            # plt.gca().invert_yaxis()
            # plt.title('y' + ':E')
            # # 子图 3：人工标注的 F1 层
            # plt.subplot(1,4,3)
            # plt.imshow(human_res[0,:ori_height,:ori_width,1])
            # plt.gca().invert_yaxis()
            # plt.title('y' + ':F1')
            # # 子图 4：人工标注的 F2 层
            # plt.subplot(1,4,4)
            # plt.imshow(human_res[0,:ori_height,:ori_width,2])
            # plt.gca().invert_yaxis()
            # plt.title('y' + ':F2')

            # # -------------------- 画图部分 4：三种结果的行列拼接对比 --------------------
            # # 使用辅助函数 plot() 分别画出 human_res / artist_res / Dias_res 的三通道图，
            # # 三行对应三种来源，列对应 E / F1 / F2，便于逐通道视觉比较
            # plt.figure(figsize=(16,16))
            # plot(human_res, 'human_res', 0)
            # plot(artist_res, 'artist_res', 1)
            # plot(Dias_res, 'Dias_res', 2)
            # # plt.close()

            # -------------------- 画图部分 5 --------------------
            # 从输入 / 人工 / 专家 / 模型 四个角度放在一行对比，并保存成 PNG 文件
            plt.figure(figsize=(18,6))
            # 子图 1：输入数据的通道
            plt.subplot(1,4,1)
            plt.imshow(test_data[0,:ori_height,:ori_width,:])
            plt.gca().invert_yaxis()
            plt.title('test_data') 
            plt.ylabel('VHSP(N)')
            plt.xlabel('CFSP(N)')
            # 子图 2：人工标注的通道
            plt.subplot(1,4,2)
            plt.imshow(human_res[0,:ori_height,:ori_width,:])
            plt.gca().invert_yaxis()
            plt.title('human_res') 
            plt.ylabel('VHSP(N)')
            plt.xlabel('CFSP(N)')
            # 子图 3：专家标注的通道
            plt.subplot(1,4,3)
            plt.imshow(artist_res[0,:ori_height,:ori_width,:])
            plt.gca().invert_yaxis()
            plt.title('artist_res')
            plt.ylabel('VHSP(N)')
            plt.xlabel('CFSP(N)')
            # 子图 4：模型预测的通道
            plt.subplot(1,4,4)
            plt.imshow(Dias_res[0,:ori_height,:ori_width,:])
            plt.gca().invert_yaxis()
            plt.title('Dias_res') 
            plt.ylabel('VHSP(N)')
            plt.xlabel('CFSP(N)')
            # 将当前这整张对比图保存到指定目录，每个样本一个文件
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
