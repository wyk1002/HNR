import numpy as np
import os
from collections import defaultdict
import pickle
import torch
import tqdm
from scipy.sparse import csc_matrix


def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.array(quadrupleList), np.asarray(times)


def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm


def get_history_target(quadruples, s_history_event_o, o_history_event_s):
    s_history_oid = []  #(total_len,[]),其中[]是历史object
    o_history_sid = []
    s_last_event = -1 * np.ones([quadruples.shape[0], 2], dtype=np.int8)
    o_last_event = -1 * np.ones([quadruples.shape[0], 2], dtype=np.int8)
    ss = quadruples[:, 0]
    rr = quadruples[:, 1]
    oo = quadruples[:, 2]

    #记录四元组对s/o的依赖次数
    s_history_related = np.zeros((quadruples.shape[0], num_e), dtype=np.int8)
    o_history_related = np.zeros((quadruples.shape[0], num_e), dtype=np.int8)
    for ix in tqdm.tqdm(range(quadruples.shape[0])):
        # s_history_oid.append([])
        # o_history_sid.append([])
        s_tmp=[]
        o_tmp=[]
        if s_history_event_o[ix]:
            s_last_event[ix] = s_history_event_o[ix][-1][-1, :]

        if o_history_event_s[ix]:
            o_last_event[ix] = o_history_event_s[ix][-1][-1, :]
        for con_events in s_history_event_o[ix]:
            idxx = (con_events[:, 0] == rr[ix]).nonzero()[0]
            cur_events = con_events[idxx, 1].tolist()
            s_tmp += con_events[:, 1].tolist()
            s_history_related[ix][cur_events] += 1
        s_history_oid.append(np.array(s_tmp,dtype=np.int8))
        for con_events in o_history_event_s[ix]:
            idxx = (con_events[:, 0] == rr[ix]).nonzero()[0]
            cur_events = con_events[idxx, 1].tolist()
            o_tmp += con_events[:, 1].tolist()
            o_history_related[ix][cur_events] += 1
        o_history_sid.append(np.array(o_tmp, dtype=np.int8))

    s_history_label_true = np.zeros((quadruples.shape[0], 1))
    o_history_label_true = np.zeros((quadruples.shape[0], 1))

    s_cnt = 0
    o_cnt = 0
    for ix in tqdm.tqdm(range(quadruples.shape[0])):
        #print('----------------------------------')
        if oo[ix] in s_history_oid[ix]:
            s_history_label_true[ix] = 1
        else:
            s_cnt += 1
            #print(oo[ix])
            #print('sssssssssssssssssss')
            #print(s_history_oid[ix])
        if ss[ix] in o_history_sid[ix]:
            o_history_label_true[ix] = 1
        else:
            o_cnt += 1
            #print(ss[ix])
            #print('oooooooooooo')
            #print(o_history_sid[ix])
    print('sss', s_cnt, 'oooo', o_cnt, s_cnt / quadruples.shape[0], o_cnt / quadruples.shape[0])
    s_history_related = csc_matrix(s_history_related)
    o_history_related = csc_matrix(o_history_related)
    return s_history_label_true, o_history_label_true, \
           s_history_related, o_history_related,\
           s_history_oid,o_history_sid,\
           s_last_event,o_last_event


graph_dict_train = {}

train_data, train_times = load_quadruples('', 'train.txt')
test_data, test_times = load_quadruples('', 'test.txt')
dev_data, dev_times = load_quadruples('', 'valid.txt')
# total_data, _ = load_quadruples('', 'train.txt', 'test.txt')

num_e, num_r = get_total_number('', 'stat.txt')

latest_t = 0
s_his = [[] for _ in range(num_e)]
o_his = [[] for _ in range(num_e)]
s_his_cache = [[] for _ in range(num_e)]
o_his_cache = [[] for _ in range(num_e)]
s_his_cache_t = [None for _ in range(num_e)]
o_his_cache_t = [None for _ in range(num_e)]
s_his_t = [[] for _ in range(num_e)]
o_his_t = [[] for _ in range(num_e)]

def dealData(tmp_data,state='train'):
    global latest_t
    s_history_data = [[] for _ in range(len(tmp_data))]
    o_history_data = [[] for _ in range(len(tmp_data))]
    s_history_data_t = [[] for _ in range(len(tmp_data))]
    o_history_data_t = [[] for _ in range(len(tmp_data))]

    # for tim in train_times:
    #     print(str(tim)+'\t'+str(max(train_times)))
    #     data = get_data_with_t(tmp_data, tim)
    #     graph_dict_train[tim] = get_big_graph(data, num_r)

    for i, train in enumerate(tmp_data):
        if i % 10000 == 0:
            print("train", i, len(tmp_data))
        # if i == 10000:
        #     break
        t = train[3]
        if latest_t != t:

            for ee in range(num_e):
                if len(s_his_cache[ee]) != 0:
                    s_his[ee].append(s_his_cache[ee].copy())
                    s_his_t[ee].append(s_his_cache_t[ee])
                    s_his_cache[ee] = []
                    s_his_cache_t[ee] = None
                if len(o_his_cache[ee]) != 0:
                    o_his[ee].append(o_his_cache[ee].copy())
                    o_his_t[ee].append(o_his_cache_t[ee])
                    o_his_cache[ee] = []
                    o_his_cache_t[ee] = None
            latest_t = t
        s = train[0]
        r = train[1]
        o = train[2]
        # print(s_his[r][s])

        s_history_data[i] = s_his[s].copy()
        o_history_data[i] = o_his[o].copy()
        s_history_data_t[i] = s_his_t[s].copy()
        o_history_data_t[i] = o_his_t[o].copy()
        # print(o_history_data_g[i])

        if len(s_his_cache[s]) == 0:
            s_his_cache[s] = np.array([[r, o]])
        else:
            s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)
        s_his_cache_t[s] = t

        if len(o_his_cache[o]) == 0:
            o_his_cache[o] = np.array([[r, s]])
        else:
            o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)
        o_his_cache_t[o] = t

        # print(s_history_data,o_history_data)

        # print(s_history_data[i], s_history_data_g[i])
        # with open('ttt.txt', 'wb') as fp:
        #     pickle.dump(s_history_data_g, fp)
        # print("save")

    '''
    s_history_data:s的邻居，不验证关系
    s_label:o在s_history_data是否有存在
    s_history_related： s,o都验证的频率,历史
    '''

    s_label_train, o_label_train, s_history_related_train, o_history_related_train, s_history_oid, o_history_sid, \
    s_last_event, o_last_event = get_history_target(tmp_data, s_history_data, o_history_data)
    with open(state+'_s_label.txt', 'wb') as fp:
        pickle.dump(s_label_train, fp)
    with open(state+'_o_label.txt', 'wb') as fp:
        pickle.dump(o_label_train, fp)
    with open(state+'_s_frequency.txt', 'wb') as fp:
        pickle.dump(s_history_related_train, fp)
    with open(state+'_o_frequency.txt', 'wb') as fp:
        pickle.dump(o_history_related_train, fp)
    with open(state+'_s_neighbor.txt', 'wb') as fp:
        pickle.dump([s_history_oid, s_last_event], fp)
    with open(state+'_o_neighbor.txt', 'wb') as fp:
        pickle.dump([o_history_sid, o_last_event], fp)

dealData(train_data,"train")
dealData(dev_data,"dev")
dealData(test_data,"test")
