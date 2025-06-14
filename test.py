# Name: test
# Author: Reacubeth
# Time: 2021/8/25 10:51
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*
import torch
import utils


def execute_test(args, total_data, model,
                 test_data,
                 test_s_label,
                 test_o_label,
                 test_s_frequency,
                 test_o_frequency,
                 test_s_history_oids,
                 test_o_history_oids,
                 test_s_last_event,
                 test_o_last_event,useCUDA):
    s_ranks1 = []
    o_ranks1 = []
    all_ranks1 = []

    s_ranks2 = []
    o_ranks2 = []
    all_ranks2 = []

    s_ranks3 = []
    o_ranks3 = []
    all_ranks3 = []
    total_data = utils.to_device(torch.from_numpy(total_data)).long()
    model.eval()
    for batch_data in utils.make_batch(test_data,
                                       test_s_label,
                                       test_o_label,
                                       test_s_frequency,
                                       test_o_frequency,
                                       test_s_history_oids,
                                       test_o_history_oids,
                                       test_s_last_event,
                                       test_o_last_event,
                                       args.batch_size):
        batch_data[0] = utils.to_device(torch.from_numpy(batch_data[0])).long()
        batch_data[1] = torch.from_numpy(batch_data[1]).float()
        batch_data[2] = torch.from_numpy(batch_data[2]).float()
        batch_data[3] = utils.to_device(torch.from_numpy(batch_data[3])).float()
        batch_data[4] = utils.to_device(torch.from_numpy(batch_data[4])).float()


        if useCUDA:
            batch_data[0] = batch_data[0].cuda()
            batch_data[1] = batch_data[1].cuda()
            batch_data[2] = batch_data[2].cuda()
            batch_data[3] = batch_data[3].cuda()
            batch_data[4] = batch_data[4].cuda()

        with torch.no_grad():
            sub_rank1, obj_rank1, cur_loss1, \
            sub_rank2, obj_rank2, cur_loss2, \
            sub_rank3, obj_rank3, cur_loss3, ce_all_acc = model(batch_data, 'Test', total_data)

            s_ranks1 += sub_rank1
            o_ranks1 += obj_rank1
            tmp1 = sub_rank1 + obj_rank1
            all_ranks1 += tmp1

            s_ranks2 += sub_rank2
            o_ranks2 += obj_rank2
            tmp2 = sub_rank2 + obj_rank2
            all_ranks2 += tmp2

            s_ranks3 += sub_rank3
            o_ranks3 += obj_rank3
            tmp3 = sub_rank3 + obj_rank3
            all_ranks3 += tmp3
    return s_ranks1, o_ranks1, all_ranks1, \
           s_ranks2, o_ranks2, all_ranks2, \
           s_ranks3, o_ranks3, all_ranks3