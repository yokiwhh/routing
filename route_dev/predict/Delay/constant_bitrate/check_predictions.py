import os
import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')

from read_dataset import input_fn, network_to_hypergraph
from model import GNN_Model
from datanetAPI import DatanetAPI
import configparser
import tensorflow as tf
import networkx as nx
import re
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def transformation(x, y):
    traffic_mean = 660.5723876953125
    traffic_std = 420.22003173828125
    packets_mean = 0.6605737209320068
    packets_std = 0.42021000385284424
    capacity_mean = 25442.669921875
    capacity_std = 16217.9072265625

    '''
    x["traffic"] = (x["traffic"] - traffic_mean) / traffic_std

    x["packets"] = (x["packets"] - packets_mean) / packets_std

    x["capacity"] = (x["capacity"] - capacity_mean) / capacity_std
    '''
    x["traffic"] = (np.log(x["traffic"] / 1000*1500))

    return x, tf.math.log(y)


def denorm_MAPE(y_true, y_pred):
    denorm_y_true = tf.math.exp(y_true)
    denorm_y_pred = tf.math.exp(y_pred)
    return tf.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100


params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('config.ini')

ds_test = input_fn('../../../data/traffic_models/constant_bitrate/test', label='delay', shuffle=False)
#ds_test = ds_test.map(lambda x, y: transformation(x, y))
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

optimizer = tf.keras.optimizers.Adam(learning_rate=float(params['HYPERPARAMETERS']['learning_rate']))

model = GNN_Model(params)

loss_object = tf.keras.losses.MeanSquaredError()

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False,
              metrics=denorm_MAPE)

best = None
best_mre = float('inf')
for f in os.listdir('./ckpt_dir'):
    if os.path.isfile(os.path.join('./ckpt_dir', f)):
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
model.load_weights('./ckpt_dir/{}'.format(best))
#model.load_weights('./ckpt_dir/01-67.74')

print("PREDICTING...")

# model.evaluate(ds_test)

predictions = model.predict(ds_test)
pred = np.squeeze(predictions)

tool = DatanetAPI('../../../data/traffic_models/constant_bitrate/test', shuffle=False)
it = iter(tool)
index = 0
num_samples = 0
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
    for path in path_nodes:
        # if HG.nodes[path]['pred_delay'] < 0 :
            # HG.nodes[path]['pred_delay'] = 0.1
        HG.nodes[path]['MRE'] = (HG.nodes[path]['pred_delay'] - HG.nodes[path]['delay']) / HG.nodes[path][
            'delay']
        print("predict throughput: %s %s ms" % (path, HG.nodes[path]['pred_delay']))
        print("original throughput: %s %s ms" % (path, HG.nodes[path]['delay']))
        l_mre.append(HG.nodes[path]['MRE'])
    index += len(path_nodes)

    print("MAPE: {:.2f} %".format(np.mean(np.abs(l_mre))))
# print(prediction)
# predictions = np.exp(predictions)

#np.save('predictions.npy', predictions)
