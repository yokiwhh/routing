import os
import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
sys.path.insert(1, './')

from read_dataset import input_fn, network_to_hypergraph
from model import GNN_Model
from datanetAPI import DatanetAPI
import configparser
import tensorflow as tf
import networkx as nx
import re
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Predict_delay():
    def __init__(self):
        self.delay_msg = 0

    def denorm_MAPE(self, y_true, y_pred):
        denorm_y_true = tf.math.exp(y_true)
        denorm_y_pred = tf.math.exp(y_pred)
        return tf.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100

    def predict(self):

        params = configparser.ConfigParser()
        params._interpolation = configparser.ExtendedInterpolation()
        params.read('config.ini')
        '''
        params['HYPERPARAMETERS']['learning_rate'] = 0.001
        params['HYPERPARAMETERS']['link_state_dim'] = 32
        params['HYPERPARAMETERS']['path_state_dim'] = 32
        params['HYPERPARAMETERS']['queue_state_dim'] = 32
        params['HYPERPARAMETERS']['t'] = 6
        params['HYPERPARAMETERS']['readout_units'] = 16
        '''


        ds_test = input_fn('/opt/DRL-OR-DEV/data/traffic_models/constant_bitrate/train', label='delay', shuffle=False)
        # ds_test = ds_test.map(lambda x, y: transformation(x, y))
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        #optimizer = tf.keras.optimizers.Adam(learning_rate=float(params['HYPERPARAMETERS']['learning_rate']))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model = GNN_Model(params)

        loss_object = tf.keras.losses.MeanSquaredError()

        model.compile(loss=loss_object,
                      optimizer=optimizer,
                      run_eagerly=False,
                      metrics=self.denorm_MAPE)

        print(sys.path)
        best = None
        best_mre = float('inf')
        for f in os.listdir('/opt/DRL-OR-DEV/predict/Delay/constant_bitrate/ckpt_dir'):
            if os.path.isfile(os.path.join('/opt/DRL-OR-DEV/predict/Delay/constant_bitrate/ckpt_dir', f)):
                reg = re.findall("\d+\.\d+", f)
                if len(reg) > 0:
                    mre = float(reg[0])
                    if mre <= best_mre:
                        best = f.replace('.index', '')
                        if '.data' in best:
                            idx = best.rfind('.')
                            best = best[:idx]
                        best_mre = mre

        print("BEST CHECKOINT FOUND: {}".format(best))
        model.load_weights('/opt/DRL-OR-DEV/predict/Delay/constant_bitrate/ckpt_dir/{}'.format(best))
        # model.load_weights('./ckpt_dir/01-67.74')

        print("PREDICTING...")

        # model.evaluate(ds_test)

        predictions = model.predict(ds_test)
        pred = np.squeeze(predictions)

        tool = DatanetAPI('/opt/DRL-OR-DEV/data/traffic_models/constant_bitrate/train', shuffle=False)
        it = iter(tool)
        index = 0
        num_samples = 0
        predict_delay = []
        for sample in it:
            num_samples += 1
            print(num_samples)

            HG = network_to_hypergraph(sample=sample)
            link_nodes = [n for n in HG.nodes if n.startswith('l_')]
            path_nodes = [n for n in HG.nodes if n.startswith('p_')]

            pred_delay = [{"pred_delay": occ} for occ in pred[index:index + len(path_nodes)]]
            delay_dict = dict(zip(path_nodes, pred_delay))
            nx.set_node_attributes(HG, delay_dict)

            l_mre = []
            l_predict = []
            l_origin = []
            for path in path_nodes:
                '''
                HG.nodes[path]['MRE'] = (HG.nodes[path]['pred_delay'] - HG.nodes[path]['delay']) / HG.nodes[path][
                    'delay']
                print("predict throughput: %s %s ms" % (path, HG.nodes[path]['pred_delay']))
                print("original throughput: %s %s ms" % (path, HG.nodes[path]['delay']))
                l_mre.append(HG.nodes[path]['MRE'])
                '''
                predict_delay.append(HG.nodes[path]['pred_delay'])
                l_predict.append(HG.nodes[path]['pred_delay'])
                l_origin.append(HG.nodes[path]['delay'])

            print("predict_sample: {:.2f} ms".format(np.mean(np.abs(l_predict))))
            print("origin_sample: {:.2f} ms".format(np.mean(np.abs(l_origin))))
            index += len(path_nodes)

        delay_msg = np.mean(np.abs(predict_delay))
        print("predict_all_sample: {:.2f} ms".format(delay_msg))
        return delay_msg
            #print("MAPE: {:.2f} %".format(np.mean(np.abs(l_mre))))