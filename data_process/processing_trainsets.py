# -*- coding: utf-8 -*-

from Processing_trainsets_synthetic import processing_trainsets_synthetic
# from dataset_processing.Processing_testsets_realscene import processing_testsets_realscene

for scale in [4, 8, 16]:
    # for dataset_name in ['Middlebury','NYU','Lu','RGBDD']:
    # if dataset_name == 'Middlebury':
    #     data_path = r'RawDatasets\Middlebury'
    #     save_path = r'DatasetsAfterProcessing\Middlebury_AfterProcessing_'+str(scale)+'X'
    dataset_name = 'NYU'
    data_path = './data/NYUDepthv2'
    save_path = './data/NYUDepthv2/depth_train_'+str(scale)+'X'
    # elif dataset_name == 'Lu':
    #     data_path = r'RawDatasets\Lu'
    #     save_path = r'DatasetsAfterProcessing\Lu_AfterProcessing_'+str(scale)+'X'
    # elif dataset_name == 'RGBDD':
    #     data_path = r'RawDatasets\RGBDD'
    #     save_path = r'DatasetsAfterProcessing\RGBDD_AfterProcessing_'+str(scale)+'X'
    processing_trainsets_synthetic(scale,data_path,save_path,dataset_name)
    print('%s dataset in %sX has been processed.'%(dataset_name,scale))


# processing_testsets_realscene(data_path = r'RawDatasets\RGBDD_Test_Realscene',
#                               save_path = r'DatasetsAfterProcessing\RGBDD_AfterProcessing_RealSceneX',
#                               dataset_name = 'RGBDD_Realscene')
# print('RGBDD_Realscene dataset has been processed.')
