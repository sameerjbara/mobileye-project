from crops_creator import get_zoom_rect
from find_tfl_attention import run_attention
from DataBase.TFLDecisionsTable import TFLDecisionsTable as d_tfl
from DataBase.TFLCoordinateTable import TFLCoordinateTable as t_tfl

import NeuralNetwork.data_utils as du
import NeuralNetwork.train_demo as td

from crop_validation import crops_validation

if __name__ == '__main__':
    run_attention()
    get_zoom_rect()
    crops_validation()
    t_tfl().export_tfls_coordinates_to_h5()
    d_tfl().export_tfls_decisions_to_h5()
    train_dataset = du.TrafficLightDataSet('Resources', 'Resources/leftImg8bit/train')
    test_dataset = du.TrafficLightDataSet('Resources', 'Resources/leftImg8bit/test', is_train=False)
    NN = du.ModelManager.make_empty_model()
    x = td.train_a_model(NN, train_dataset, test_dataset, log_dir='Resources/log_dir', num_epochs=50)
    td.examine_my_results("Resources", 'Resources/leftImg8bit/train', 'Resources/log_dir/model.pkl', test_dataset)
