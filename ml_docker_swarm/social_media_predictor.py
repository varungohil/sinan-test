# Note: must run from microservices directory!
import sys
import os
import socket
import subprocess
import time
import json
import math
import random
import argparse,logging

import mxnet as mx
import xgboost as xgb
import numpy as np
from importlib import import_module

# ml parameters
Model = None
InternalSysState = None
BoostTree = None
Services   = ['compose-post-redis',
              'compose-post-service',
              'home-timeline-redis',
              'home-timeline-service',
              # 'jaeger',
              'nginx-thrift',
              'post-storage-memcached',
              'post-storage-mongodb',
              'post-storage-service',
              'social-graph-mongodb',
              'social-graph-redis',
              'social-graph-service',
              'text-service',
              'text-filter-service',
              'unique-id-service',
              'url-shorten-service',
              'media-service',
              'media-filter-service',
              'user-mention-service',
              'user-memcached',
              'user-mongodb',
              'user-service',
              'user-timeline-mongodb',
              'user-timeline-redis',
              'user-timeline-service',
              'write-home-timeline-service',
              'write-home-timeline-rabbitmq',
              'write-user-timeline-service',
              'write-user-timeline-rabbitmq']

# -----------------------------------------------------------------------
# parser args definition
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('--cpus', dest='cpus', type=int, required=True)
# parser.add_argument('--stack-name', dest='stack_name', type=str, required=True)
parser.add_argument('--cnn-time-steps', dest='cnn_time_steps', type=int, default=5)
# xgb-look-forward doesn't include immediate future (next time step)
parser.add_argument('--xgb-look-forward', dest='xgb_look_forward', type=int, default=4)
parser.add_argument('--server-port', dest='server_port', type=int, default=40010)
parser.add_argument('--model-prefix', dest='model_prefix', type=str, default='./model/cnv')
parser.add_argument('--load-epoch', dest='load_epoch', type=int, default=200)
parser.add_argument('--xgb-prefix', dest='xgb_prefix', type=str, 
    default='./xgb_model/social_nn_sys_state_look_forward_')
parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')

args = parser.parse_args()

CnnTimeSteps = args.cnn_time_steps
XgbLookForward = args.xgb_look_forward
ServerPort = args.server_port		 

def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def _compose_sys_data_channel(sys_data, field, batch_size):
    global Services
    global CnnTimeSteps

    for i, service in enumerate(Services):
        assert len(sys_data[service][field]) == CnnTimeSteps
        if i == 0:
            data = np.array(sys_data[service][field])
        else:
            data = np.vstack((data, np.array(sys_data[service][field])))

    data = data.reshape([1, data.shape[0], data.shape[1]])
    for i in range(0, batch_size):
        if i == 0:
            channel_data = np.array(data)
        else:
            channel_data = np.vstack((channel_data, data))
    channel_data = channel_data.reshape([channel_data.shape[0], 1, channel_data.shape[1], channel_data.shape[2]])
    return channel_data

def _predict(info, uncertainty_threshold=0.1):
    global Services
    global Model
    global InternalSysState
    global BoostTree
    global CnnTimeSteps
    global XgbLookForward

    raw_sys_data = info['sys_data']
    raw_next_info = info['next_info']
    batch_size = len(raw_next_info)

    # rps
    rps_data = _compose_sys_data_channel(raw_sys_data, 'rps', batch_size)

    # replica
    replica_data = _compose_sys_data_channel(raw_sys_data, 'replica', batch_size)

    # cpu limit
    cpu_limit_data = _compose_sys_data_channel(raw_sys_data, 'cpu_limit', batch_size)

    # cpu usage
    cpu_usage_mean_data = _compose_sys_data_channel(raw_sys_data, 'cpu_usage_mean', batch_size)

    # memory
    rss_mean_data = _compose_sys_data_channel(raw_sys_data, 'rss_mean', batch_size)

    cache_mem_mean_data = _compose_sys_data_channel(raw_sys_data, 'cache_mem_mean', batch_size)

    sys_data = np.concatenate(
        (rps_data, 
         replica_data, 
         cpu_limit_data,
         cpu_usage_mean_data,
         # memory
         rss_mean_data,
         cache_mem_mean_data), 
        axis=1)

    logging.info('sys_data.shape = ' + str(sys_data.shape))

    #-------------------------- e2e_lat --------------------------#
    for key in ['90.0', '95.0', '98.0', '99.0', '99.9']:
        assert len(raw_sys_data['e2e_lat'][key]) == CnnTimeSteps
        if key == '90.0':
            e2e_lat = np.array(raw_sys_data['e2e_lat'][key])
        else:
            e2e_lat = np.vstack((e2e_lat, np.array(raw_sys_data['e2e_lat'][key])))

    # print 'e2e_lat ori shape = ', e2e_lat.shape
    e2e_lat = e2e_lat.reshape([1, e2e_lat.shape[0], e2e_lat.shape[1]])

    for i in range(0, batch_size):
        if i == 0:
            lat_data = np.array(e2e_lat)
        else:
            lat_data = np.vstack((lat_data, e2e_lat))

    logging.info('lat_data.shape = ' + str(lat_data.shape))

    #-------------------------- next_info --------------------------#
    ncore_next = None
    ncore_next_k = None
    for i, proposal in enumerate(raw_next_info):
        for j, service in enumerate(Services):
            if j == 0:
                ncore_proposal = np.array(proposal[service]['cpus'])
            else:
                ncore_proposal = np.vstack((ncore_proposal, np.array(proposal[service]['cpus'])))

        for k in range(0, XgbLookForward):
            if k == 0:
                ncore_proposal_next_k = np.array(ncore_proposal).reshape([-1, 1])
            else:
                ncore_proposal_next_k = np.hstack((ncore_proposal_next_k, 
                                            np.array(ncore_proposal).reshape([-1, 1])))

        if i == 0:
            ncore_next = ncore_proposal.reshape([1, ncore_proposal.shape[0]])
            ncore_next_k = ncore_proposal_next_k.reshape(
                            [1, ncore_proposal_next_k.shape[0], ncore_proposal_next_k.shape[1]])
        else:
            ncore_next = np.vstack((ncore_next, ncore_proposal.reshape([1, ncore_proposal.shape[0]])))
            ncore_next_k = np.vstack((ncore_next_k, 
                        ncore_proposal_next_k.reshape(
                            [1, ncore_proposal_next_k.shape[0], ncore_proposal_next_k.shape[1]])))

    next_data = ncore_next
    next_k_data = ncore_next_k

    logging.info('next_data.shape = ' + str(next_data.shape))
    logging.info('next_k_data.shape = ' + str(next_k_data.shape))

    pred_data  = {'data1':sys_data, 'data2':lat_data, 'data3':next_data}
    pred_iter = mx.io.NDArrayIter(pred_data, batch_size=batch_size)
    cnn_pred = Model.predict(pred_iter).asnumpy()

    #-------------------- predicting next_k cycle with xgb --------------------#
    internal_sys_state = InternalSysState.predict(pred_iter).asnumpy()
    next_k_info = next_k_data.reshape(next_k_data.shape[0], -1)
    xgb_input = np.concatenate((internal_sys_state, next_k_info), axis = 1)
    dpred = xgb.DMatrix(xgb_input)
    xgb_predict = BoostTree.predict(dpred)

    uncertainty_std = np.std(xgb_predict) / (np.sqrt(batch_size) + 1e-8)
    uncertainty_entropy = -np.sum(xgb_predict * np.log(xgb_predict + 1e-8)) / batch_size
    uncertainty_variance = np.var(xgb_predict)

    uncertainty = max(uncertainty_std, uncertainty_entropy, uncertainty_variance)
    predict = []
    for i in range(batch_size):
        if uncertainty > uncertainty_threshold:
            predict.append([-1, -1])
        else:
            predict.append([round(cnn_pred[i, -2], 2), round(xgb_predict[i], 3)])

    return predict

def test():
    global Model
    global InternalSysState
    global BoostTree
    global ServerPort
    global CnnTimeSteps
    global XgbLookForward

    # load model for prediction
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    load_params = _load_model(args, kv.rank)
    sym = load_params[0]
    all_layers = sym.get_internals()

    #---------------- cnn -----------------#
    Model   = all_layers['latency_output']
    Model   = mx.mod.Module(
            context = devs,
            symbol = Model,
            # symbol = sym,
            data_names = ('data1','data2', 'data3'),
            # label_names = ('label',)
            )

    default_batch_size = 2048
    Model.bind(for_training=False, 
        data_shapes=[('data1', (default_batch_size,6,28,CnnTimeSteps)), 
                     ('data2', (default_batch_size,5,CnnTimeSteps)), 
                     # ('data3', (default_batch_size,2,28))
                     ('data3', (default_batch_size,28))])

    Model.set_params(load_params[1], load_params[2], allow_missing=True, allow_extra=True)

    #---------------- xgb -----------------#
    # model's internal representation
    all_layers = sym.get_internals()
    InternalSysState = all_layers['full_feature_output']
    InternalSysState   = mx.mod.Module(
            context = devs,
            symbol = InternalSysState,
            # symbol = sym,
            data_names = ('data1','data2', 'data3')
            # label_names = ('label',)
            )
    InternalSysState.bind(for_training=False, 
        data_shapes=[('data1', (default_batch_size,6,28,CnnTimeSteps)), 
                     ('data2', (default_batch_size,5,CnnTimeSteps)), 
                     # ('data3', (default_batch_size,2,28))
                     ('data3', (default_batch_size,28))])
    InternalSysState.set_params(load_params[1], load_params[2], allow_missing=True, allow_extra=True)

    BoostTree = xgb.Booster()  # init model
    print 'load ', args.xgb_prefix + str(XgbLookForward) + '.model'
    BoostTree.load_model(args.xgb_prefix + str(XgbLookForward) + '.model')  # load data

    info = {}
    sys_data = {}
    sys_data['e2e_lat']  = {}
    # info['rps_next'] = 2000
    for i, key in enumerate(['90.0', '95.0', '98.0', '99.0', '99.9']):
        sys_data['e2e_lat'][key] = [1.0 + i/10.0] * CnnTimeSteps

    for service in Services:
        sys_data[service] = {}
        sys_data[service]['rps'] =  [50] * CnnTimeSteps
        sys_data[service]['cpu_limit'] = [12] * CnnTimeSteps
        sys_data[service]['replica'] = [10] * CnnTimeSteps
        sys_data[service]['cpu_usage_mean'] = [5.0] * CnnTimeSteps
        sys_data[service]['rss_mean']  = [1.0]  * CnnTimeSteps
        sys_data[service]['cache_mem_mean'] = [0.0] * CnnTimeSteps

    info['sys_data'] = sys_data
    next_info = []
    batch_size = 900
    for i in range(0, batch_size):
        proposal = {}
        for service in Services:
            proposal[service] = {}
            proposal[service]['cpus'] = 12
            proposal[service]['rps'] = 50
        next_info.append(proposal)
        # info[service]['read_req_num_next']  = 2000
        # info[service]['write_req_num_next'] = 300
    info['next_info'] = next_info

    t_s = time.time()
    pred = _predict(info)
    print 'inf time: ', time.time() - t_s
    # print pred

def main():
    global Model
    global InternalSysState
    global BoostTree
    global ServerPort
    global CnnTimeSteps
    global XgbLookForward

    # load model for prediction
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    load_params = _load_model(args, kv.rank)
    sym = load_params[0]
    all_layers = sym.get_internals()

    #---------------- cnn -----------------#
    Model   = all_layers['latency_output']
    Model   = mx.mod.Module(
            context = devs,
            symbol = Model,
            # symbol = sym,
            data_names = ('data1','data2', 'data3'),
            # label_names = ('label',)
            )

    default_batch_size = 2048
    Model.bind(for_training=False, 
        data_shapes=[('data1', (default_batch_size,6,28,CnnTimeSteps)), 
                     ('data2', (default_batch_size,5,CnnTimeSteps)), 
                     # ('data3', (default_batch_size,2,28))
                     ('data3', (default_batch_size,28))])

    Model.set_params(load_params[1], load_params[2], allow_missing=True, allow_extra=True)

    #---------------- xgb -----------------#
    # model's internal representation
    all_layers = sym.get_internals()
    InternalSysState = all_layers['full_feature_output']
    InternalSysState   = mx.mod.Module(
            context = devs,
            symbol = InternalSysState,
            # symbol = sym,
            data_names = ('data1','data2', 'data3')
            # label_names = ('label',)
            )
    InternalSysState.bind(for_training=False, 
        data_shapes=[('data1', (default_batch_size,6,28,CnnTimeSteps)), 
                     ('data2', (default_batch_size,5,CnnTimeSteps)), 
                     # ('data3', (default_batch_size,2,28))
                     ('data3', (default_batch_size,28))])
    InternalSysState.set_params(load_params[1], load_params[2], allow_missing=True, allow_extra=True)

    BoostTree = xgb.Booster()  # init model
    logging.info('load ' + args.xgb_prefix + str(XgbLookForward) + '.model')
    BoostTree.load_model(args.xgb_prefix + str(XgbLookForward) + '.model')  # load data

    logging.info('model loaded...')

    local_serv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    local_serv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    local_serv_sock.bind(('0.0.0.0', ServerPort))
    local_serv_sock.listen(1024)
    host_sock, addr = local_serv_sock.accept()
    host_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    logging.info('master connected')

    MsgBuffer = ''
    terminate = False

    while True:
        data = host_sock.recv(2048).decode('utf-8')
        if len(data) == 0:
            logging.warning('connection reset by host, exiting...')
            break
        # else:
        #	 print 'recv in main loop: ' + data

        MsgBuffer += data
        while '\n' in MsgBuffer:
            (cmd, rest) = MsgBuffer.split('\n', 1)
            MsgBuffer = rest

            # print 'cmd = ', cmd
            if cmd.startswith('pred----'):
                info = json.loads(cmd.split('----')[-1])
                pred_lat = _predict(info)
                # print pred_lat
                ret_msg = 'pred----' + json.dumps(pred_lat) + '\n'
                host_sock.sendall(ret_msg.encode('utf-8'))

            elif cmd.startswith('terminate'):
                ret_msg = 'experiment_done\n'
                host_sock.sendall(ret_msg.encode('utf-8'))
                terminate = True
                break

            else:
                logging.error('Unknown cmd format')
                logging.error(cmd) 
                terminate = True
                break

        if terminate:
            break
    
    host_sock.close()
    local_serv_sock.close()
  
if __name__ == "__main__":	
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    main()
    # test()
