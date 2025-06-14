# Name: valid
# Author: Reacubeth
# Time: 2021/8/25 10:30
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import torch
import utils


def execute_valid(args, total_data, model,
                  dev_data,
                  dev_s_label,
                  dev_o_label,
                  dev_s_frequency,
                  dev_o_frequency,
                  dev_s_history_oids,
                  dev_o_history_oids,
                  dev_s_last_event,
                  dev_o_last_event,useCUDA):
    s_ranks2 = []
    o_ranks2 = []
    all_ranks2 = []

    s_ranks3 = []
    o_ranks3 = []
    all_ranks3 = []
    total_data = utils.to_device(torch.from_numpy(total_data)).long()
    model.eval()
    for batch_data in utils.make_batch(dev_data,
                                       dev_s_label,
                                       dev_o_label,
                                       dev_s_frequency,
                                       dev_o_frequency,
                                       dev_s_history_oids,
                                       dev_o_history_oids,
                                       dev_s_last_event,
                                       dev_o_last_event,
                                       args.batch_size):
        batch_data[0] = torch.from_numpy(batch_data[0]).long()
        batch_data[1] = torch.from_numpy(batch_data[1]).float()
        batch_data[2] = torch.from_numpy(batch_data[2]).float()
        batch_data[3] = torch.from_numpy(batch_data[3]).float()
        batch_data[4] = torch.from_numpy(batch_data[4]).float()

        if useCUDA:
            batch_data[0] = batch_data[0].cuda()
            batch_data[1] = batch_data[1].cuda()
            batch_data[2] = batch_data[2].cuda()
            batch_data[3] = batch_data[3].cuda()
            batch_data[4] = batch_data[4].cuda()

        with torch.no_grad():
            _, _, _, \
            sub_rank2, obj_rank2, cur_loss2, \
            sub_rank3, obj_rank3, cur_loss3, ce_all_acc = model(batch_data, 'Valid', total_data)

            s_ranks2 += sub_rank2
            o_ranks2 += obj_rank2
            tmp2 = sub_rank2 + obj_rank2
            all_ranks2 += tmp2

            s_ranks3 += sub_rank3
            o_ranks3 += obj_rank3
            tmp3 = sub_rank3 + obj_rank3
            all_ranks3 += tmp3
    return s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3