# -*- coding: utf-8 -*-

from Processing_testsets_synthetic import processing_testsets_synthetic
# from dataset_processing.Processing_testsets_realscene import processing_testsets_realscene

for scale in [4, 8, 16]:
    for dataset_name in ['Middlebury','NYU','Lu','RGBDD']:
        if dataset_name == 'Middlebury':
            data_path = './data/GDSR_test/Middlebury'
            save_path = './data/RGBD_test/test_AfterProcessing_'+str(scale)+'X'+'/Middlebury'
        elif dataset_name == 'NYU':
            data_path = './data/GDSR_test/NYU'
            save_path = './data/RGBD_test/test_AfterProcessing_'+str(scale)+'X'+'/NYU'
        elif dataset_name == 'Lu':
            data_path = './data/GDSR_test/Lu'
            save_path = './data/RGBD_test/test_AfterProcessing_'+str(scale)+'X'+'/Lu'
        elif dataset_name == 'RGBDD':
            data_path = './data/GDSR_test/RGBDD'
            save_path = './data/RGBD_test/test_AfterProcessing_'+str(scale)+'X'+'/RGBDD'
        processing_testsets_synthetic(scale,data_path,save_path,dataset_name)
        print('%s dataset in %sX has been processed.'%(dataset_name,scale))


# processing_testsets_realscene(data_path = r'RawDatasets\RGBDD_Test_Realscene',
#                               save_path = r'DatasetsAfterProcessing\RGBDD_AfterProcessing_RealSceneX',
#                               dataset_name = 'RGBDD_Realscene')
# print('RGBDD_Realscene dataset has been processed.')
