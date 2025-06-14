# Name: main
# Author: Reacubeth
# Time: 2021/6/25 17:05
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import argparse
import datetime
import os
import pickle
import time
import csv
import numpy as np
import torch

import utils
import test
import valid


def main_portal(args):

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('***************GPU_ID***************: ', 1)
    else:
        # raise NotImplementedError
        pass
    seed=999
    np.random.seed(seed)
    torch.manual_seed(seed)
    settings = {}

    test_sub = '/test_history_sub.txt'
    test_ob = '/test_history_ob.txt'
    test_s_label_f = '/test_s_label.txt'
    test_o_label_f = '/test_o_label.txt'
    test_s_frequency_f = '/test_s_frequency.txt'
    test_o_frequency_f = '/test_o_frequency.txt'

    test_data, _ = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
    try:
        total_data, _ = utils.load_quadruples('./data/' + args.dataset, 'train.txt', 'valid.txt', 'test.txt')
    except:
        total_data, _ = utils.load_quadruples('./data/' + args.dataset, 'train.txt', 'test.txt')
        print(args.dataset, 'does not have valid set.')
    total_data=utils.to_device(torch.from_numpy(total_data))

    with open('data/' + args.dataset + test_sub, 'rb') as f:
        s_history_test_data = pickle.load(f)
    with open('data/' + args.dataset + test_ob, 'rb') as f:
        o_history_test_data = pickle.load(f)
    with open('./data/' + args.dataset + test_s_label_f, 'rb') as f:
        test_s_label = pickle.load(f)
    with open('./data/' + args.dataset + test_o_label_f, 'rb') as f:
        test_o_label = pickle.load(f)
    with open('./data/' + args.dataset + test_s_frequency_f, 'rb') as f:
        test_s_frequency = pickle.load(f).toarray()
    with open('./data/' + args.dataset + test_o_frequency_f, 'rb') as f:
        test_o_frequency = pickle.load(f).toarray()

    s_history_test = s_history_test_data[0]
    o_history_test = o_history_test_data[0]


    if not args.only_eva:
        pass
    else:
        model = torch.load(args.model_dir)
        model.args=args

        no_his_num, his_repeat_num, his_new_num = 0, 0, 0

        no_his_ranks,his_repeat_ranks,his_new_ranks=[],[],[]


        # model.eval()
        for batch_data in utils.make_batch(test_data,
                                           s_history_test,
                                           o_history_test,
                                           test_s_label,
                                           test_o_label,
                                           test_s_frequency,
                                           test_o_frequency,
                                           args.batch_size):
            batch_data[0] = torch.from_numpy(batch_data[0]).long()
            batch_data[3] = torch.from_numpy(batch_data[3]).float()
            batch_data[4] = torch.from_numpy(batch_data[4]).float()
            batch_data[5] = torch.from_numpy(batch_data[5]).float()
            batch_data[6] = torch.from_numpy(batch_data[6]).float()
            if use_cuda:
                batch_data[0] = batch_data[0].cuda()
                batch_data[3] = batch_data[3].cuda()
                batch_data[4] = batch_data[4].cuda()
                batch_data[5] = batch_data[5].cuda()
                batch_data[6] = batch_data[6].cuda()
            sub_rank,obj_rank,l1,s2,o2,l2,s3, o3,l3, acc = model(batch_data, 'Test', total_data)
            # print(batch_data[0],sub_rank,obj_rank)
            # sub_rank=sub_rank.tolist()
            # obj_rank=obj_rank.tolist()
            for idx_batch in range(batch_data[0].shape[0]):
                s,r,o,t=batch_data[0][idx_batch].tolist()
                #predict obj
                if batch_data[1][idx_batch] is not None:
                    if batch_data[5][idx_batch].tolist()[o]>0:
                        #history_repeat
                        his_repeat_num+=1
                        his_repeat_ranks.append(obj_rank[idx_batch])
                    else:
                        his_new_num+=1
                        his_new_ranks.append(obj_rank[idx_batch])
                else:
                    print(s,r,o,t,obj_rank[idx_batch])
                    no_his_num+=1
                    no_his_ranks.append(obj_rank[idx_batch])

                #predict sub
                if batch_data[2][idx_batch] is not None:
                    if batch_data[6][idx_batch].tolist()[s] > 0:
                        # history_repeat
                        his_repeat_num += 1
                        his_repeat_ranks.append(sub_rank[idx_batch])
                    else:
                        his_new_num += 1
                        his_new_ranks.append(sub_rank[idx_batch])
                else:
                    print(s, r, o, t, sub_rank[idx_batch])
                    no_his_num += 1
                    no_his_ranks.append(sub_rank[idx_batch])

        print(no_his_num,his_repeat_num,his_new_num)
        print(len(no_his_ranks),len(his_repeat_ranks),len(his_new_ranks))

        no_his_ranks = np.asarray(no_his_ranks)
        s_mr_lk = np.mean(no_his_ranks)
        s_mrr_lk = np.mean(1.0 / no_his_ranks)

        print("no_his_ranks MRR (lk): {:.6f}".format(s_mrr_lk))
        print("no_his_ranks MR (lk): {:.6f}".format(s_mr_lk))
        for hit in [1, 3, 10]:
            avg_count_sub_lk = np.mean((no_his_ranks <= hit))
            print("no_his_ranks Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk))

        his_repeat_ranks = np.asarray(his_repeat_ranks)
        s_mr_lk = np.mean(his_repeat_ranks)
        s_mrr_lk = np.mean(1.0 / his_repeat_ranks)

        print("his_repeat_ranks MRR (lk): {:.6f}".format(s_mrr_lk))
        print("his_repeat_ranks MR (lk): {:.6f}".format(s_mr_lk))
        for hit in [1, 3, 10]:
            avg_count_sub_lk = np.mean((his_repeat_ranks <= hit))
            print("his_repeat_ranks Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk))

        his_new_ranks = np.asarray(his_new_ranks)
        s_mr_lk = np.mean(his_new_ranks)
        s_mrr_lk = np.mean(1.0 / his_new_ranks)

        print("his_new_ranks MRR (lk): {:.6f}".format(s_mrr_lk))
        print("his_new_ranks MR (lk): {:.6f}".format(s_mr_lk))
        for hit in [1, 3, 10]:
            avg_count_sub_lk = np.mean((his_new_ranks <= hit))
            print("his_new_ranks Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CENET')
    parser.add_argument("--description", type=str, default='your_description_for_folder_name', help="description")
    parser.add_argument("-d", "--dataset", type=str, default='GDELT', help="dataset")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=30, help="maximum epochs")
    parser.add_argument("--oracle-epochs", type=int, default=20, help="maximum oracle epochs")
    parser.add_argument("--valid-epochs", type=int, default=5, help="validation epochs")
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha for nceloss")
    parser.add_argument("--lambdax", type=float, default=2, help="lambda")

    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--oracle_lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--oracle_mode", type=str, default='soft', help="soft and hard mode for Oracle")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument("--filtering", type=utils.str2bool, default=True)

    parser.add_argument("--only_oracle", type=utils.str2bool, default=False, help="whether only to train oracle")
    parser.add_argument("--only_eva", type=utils.str2bool, default=True, help="whether only evaluation on test set")
    parser.add_argument("--model_dir", type=str,
                        default="SAVE/your_description_for_folder_name27-04-2025,10-04-02GDELT-EPOCH30/models/GDELT_best.pth",
                        help="model directory")
    parser.add_argument("--save_dir", type=str, default="SAVE", help="save directory")
    parser.add_argument("--eva_dir", type=str, default="SAVE", help="saved dir of the testing model")
    args_main = parser.parse_args()
    print(args_main)
    if not os.path.exists(args_main.save_dir):
        os.makedirs(args_main.save_dir)
    main_portal(args_main)
