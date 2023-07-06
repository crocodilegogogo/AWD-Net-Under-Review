import os
import argparse

# hyper-parameters
def parse_args():
    # The training options
      parser = argparse.ArgumentParser(description='AWD_Net')
      
      parser.add_argument('--PATTERN', type=str, default='TRAIN',
                          help='pattern: TRAIN, TEST')
      parser.add_argument('--DATASETS', nargs='+', default=['SEU'],
                          help='dataset name: CWRU_10, XJTU, SEU')
      parser.add_argument('--CLASSIFIERS_all', nargs='+', default=['AWD_Net'],
                          help='classifier name: Resnet18, WDCNN, MCNN_LSTM, DRSN_CW, LiftingNet, WaveletKernelNet, Wavelet_SANet')
      parser.add_argument('--BATCH_SIZE', type=int, default=64,
                          help='training batch size: 64')
      parser.add_argument('--EPOCH', type=int, default=200,
                          help='training epoches: 200')
      parser.add_argument('--LR', type=float, default=0.01,
                          help='learning rate: 0.01')
      parser.add_argument('--CV_SPLITS', type=int, default=5,
                          help='CV_SPLITS: 5')
      parser.add_argument('--test_split', type=int, default=1,
                          help='the testing dataset is seperated into test_split pieces in the inference process')
      
      args = parser.parse_args()
      return args

def get_CWRU_dataset_param(CUR_DIR, dataset_name):
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = filepath + '//datasets//CWRU_10//'
    MODELS_COMP_LOG_DIR = CUR_DIR + '//logs//' + dataset_name + '//classifiers_comparison1//'
    Fault_LABELS = ["Normal","7Ball", "7innerRace", "7outerRace6",
                  "14Ball", "14innerRace", "14outerRace6",
                  "21Ball", "21innerRace", "21outerRace6", ]
    FaultID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    nb_classes = len(FaultID)
    WINDOW_SIZE = 1024
    OVERLAP = 1024  # the overlap of siliding window
    INPUT_CHANNEL = 1
    SNR = 5

    return DATA_DIR, MODELS_COMP_LOG_DIR, Fault_LABELS, FaultID, \
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes

def get_SEU_dataset_param(CUR_DIR, dataset_name):
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = filepath + '//datasets//SEU//'
    MODELS_COMP_LOG_DIR = CUR_DIR + '//logs//' + dataset_name + '//classifiers_comparison//'
    Fault_LABELS = ["Chipped_20_0","Chipped_30_2",
                    "Health_20_0", "Health_30_2",
                    "Miss_20_0", "Miss_30_2",
                    "Root_20_0", "Root_30_2",
                    "Surface_20_0"]
    FaultID = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    nb_classes = len(FaultID)
    WINDOW_SIZE = 1024
    OVERLAP = 1024
    INPUT_CHANNEL = 1
    SNR = 5

    return DATA_DIR, MODELS_COMP_LOG_DIR, Fault_LABELS, FaultID, \
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes

def get_XJTU_dataset_param(CUR_DIR, dataset_name):
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = filepath + '//datasets//XJTU//'
    MODELS_COMP_LOG_DIR = CUR_DIR + '//logs//' + dataset_name + '//classifiers_comparison//'
    Fault_LABELS = ["Bearing 1_1", "Bearing 1_2",
                    "Bearing 1_3", "Bearing 1_4",
                    "Bearing 1_5",
                    "Bearing 2_1", "Bearing 2_2",
                    "Bearing 2_3", "Bearing 2_4",
                    "Bearing 2_5",
                    "Bearing 3_1", "Bearing 3_2",
                    "Bearing 3_3", "Bearing 3_4",
                    "Bearing 3_5",
                   ]
    FaultID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    nb_classes = len(FaultID)
    WINDOW_SIZE = 1024
    OVERLAP = 1024
    INPUT_CHANNEL = 1
    SNR = 5
    return DATA_DIR, MODELS_COMP_LOG_DIR, Fault_LABELS, FaultID, \
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes


def create_classifier(dataset_name, classifier_name, input_channel,
                      data_length, nb_classes, INPUT_CHANNEL):

##################################
###### comparison models

    if classifier_name == 'Resnet18':
        from classifiers.compare import Resnet18

        return Resnet18.resnet18(nb_classes, pretrained=False), Resnet18

    if classifier_name == 'WDCNN':
        from classifiers.compare import WDCNN

        return WDCNN.WDCNN(3,nb_classes), \
               WDCNN

    if classifier_name == 'MCNN_LSTM':
        from classifiers.compare import MCNN_LSTM

        return MCNN_LSTM.MCNN_LSTM(nb_classes), \
               MCNN_LSTM

    if classifier_name == 'DRSN_CW':
        from classifiers.compare import DRSN_CW
        return DRSN_CW.resnet18(nb_classes), DRSN_CW

    if classifier_name == 'LiftingNet':
        from classifiers.compare import LiftingNet

        return LiftingNet.LiftingNet(1, 3, nb_classes), \
               LiftingNet

    if classifier_name == 'WaveletKernelNet':
        from classifiers.compare import WaveletKernelNet

        return WaveletKernelNet.waveletkernelnet(nb_classes), \
               WaveletKernelNet

    if classifier_name == 'Wavelet_SANet':
        from classifiers.compare import Wavelet_SANet

        return Wavelet_SANet._4dwt_4(1, 9, 32, 32, 0.05, nb_classes, 1, 'db4'), \
               Wavelet_SANet

    if classifier_name == 'AWD_Net':
        from classifiers.compare import AWD_Net

        return AWD_Net.DAWN_Gumble(dataset_name, nb_classes, data_length, first_conv_chnnl=16,
                                               kernel_size=3, no_bootleneck=False, average_mode="mode2",
                                               classifier="mode2", share_weights=False, simple_lifting=False,
                                               COLOR=True, regu_details=0.01, regu_approx=0.01, haar_wavelet=False), \
               AWD_Net
