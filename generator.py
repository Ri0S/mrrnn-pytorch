import numpy as np
import copy

from modules import *
from util import *
from dataloader import *

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
np.random.seed(123)
if use_cuda:
    torch.cuda.manual_seed(123)


def generate(model, ses_encoding, config):
    diversity_rate = 2
    antilm_param = 10
    beam = config.beam

    n_candidates, final_candids = [], []
    candidates = [([1], 0, 0)]
    gen_len, max_gen_len = 1, 100

    while gen_len <= max_gen_len:
        for c in candidates:
            seq, pts_score, pt_score = c[0], c[1], c[2]
            _target = torch.tensor([seq], device=config.device, dtype=torch.int64)
            dec_o, dec_lm = model.dec([ses_encoding, _target, [len(seq)]])
            dec_o = dec_o[:, :, :-1]

            op = F.log_softmax(dec_o, 2, 5)
            op = op[:, -1, :]
            topval, topind = op.topk(beam, 1)

            for i in range(beam):
                ctok, cval = topind.data[0, i], topval.data[0, i]
                if ctok == 2:
                    list_to_append = final_candids
                else:
                    list_to_append = n_candidates

                list_to_append.append((seq + [ctok], pts_score + cval - diversity_rate * (i + 1), pt_score + uval))

        n_candidates.sort(key=lambda temp: sort_key(temp, config.mmi), reverse=True)
        candidates = copy.copy(n_candidates[:beam])
        n_candidates[:] = []
        gen_len += 1

    final_candids = final_candids + candidates
    final_candids.sort(key=lambda temp: sort_key(temp, config.mmi), reverse=True)

    return final_candids[:beam]


def inference_beam(dataloader, model, inv_dict, config):
    criteria = nn.CrossEntropyLoss(ignore_index=0, size_average=False)
    if use_cuda:
        criteria.cuda()

    cur_tc = model.dec.get_teacher_forcing()
    model.dec.set_teacher_forcing(True)
    fout = open(config.name + "_beam" + str(config.beam) + "_result.txt", 'w', encoding='utf-8')
    model.load_state_dict(torch.load(config.name))

    model.eval()

    for i_batch, sample_batch in enumerate(dataloader):

        testData = sample_batch[:-1]
        gold = sample_batch[-1]
        us = testData.transpose(0, 1).contiguous().clone()
        qu_seq = model.base_enc(us.view(-1, us.size(2))).view(us.size(0), us.size(1), -1)
        final_session_o = model.ses_enc(qu_seq, None)

        for k in range(testData.size(1)):
            sent = generate(model, final_session_o[k, :, :].unsqueeze(0), config)
            pt = tensor_to_sent(sent, inv_dict)
            gt = tensor_to_sent(gold[k, :].unsqueeze(0).data.cpu().numpy(), inv_dict, True)
            if not config.pretty:
                print(pt)
                print("Ground truth {} {} \n".format(gt, get_sent_ll(gold[k, :].unsqueeze(0), model, criteria,
                                                                     final_session_o)))
            else:
                for i in range(testData.size(0)):
                    t = tensor_to_sent(testData[i][k, :].unsqueeze(0).data.cpu().numpy(), inv_dict, True)
                    print(t)
                    fout.write(str(t[0]) + '\n')

                print("gold:", gt[0])
                fout.write("gold: " + str(gt[0]) + '\n')
                print("pred:", pt[0][0])
                fout.write("pred: " + str(pt[0][0]) + '\n\n')

                print()

    model.dec.set_teacher_forcing(cur_tc)
    fout.close()


def rec_inference_beam(dataloader, model, inv_dict, config):
    criteria = nn.CrossEntropyLoss(ignore_index=0, size_average=False)
    if use_cuda:
        criteria.cuda()

    cur_tc = model.dec.get_teacher_forcing()
    model.dec.set_teacher_forcing(True)
    fout = open(config.name + "_result5.txt", 'w', encoding='utf-8')
    model.load_state_dict(torch.load(config.name))
    model.eval()

    for i_batch, sample_batch in enumerate(dataloader):
        if sample_batch.size(0) < 1:
            continue
        testData = sample_batch[:1]
        gold = sample_batch[-1]

        us = testData.transpose(0, 1).contiguous().clone()
        qu_seq = model.base_enc(us.view(-1, us.size(2))).view(us.size(0), us.size(1), -1)
        final_session_o = model.ses_enc(qu_seq, None)

        for k in range(testData.size(1)):
            sent = generate(model, final_session_o[k, :, :].unsqueeze(0), config)
            u =torch.tensor(sent[0][0], device=config.device, dtype=torch.int64).view(1, 1, -1)
            quseq = model.base_enc(u.view(1, -1)).view(1, 1, -1)
            final_session_o = model.ses_enc(quseq, None)

            sent2 = generate(model, final_session_o[k, :, :].unsqueeze(0), config)

            pt = tensor_to_sent(sent, inv_dict)
            pt2 = tensor_to_sent(sent2, inv_dict)
            gt = tensor_to_sent(gold[k, :].unsqueeze(0).data.cpu().numpy(), inv_dict, True)

            if not config.pretty:
                print(pt)
                print("Ground truth {} {} \n".format(gt, get_sent_ll(gold[k, :].unsqueeze(0), model, criteria,
                                                                     final_session_o)))
            else:
                for i in range(1):
                    t = tensor_to_sent(testData[i][k, :].unsqueeze(0).data.cpu().numpy(), inv_dict, True)
                    print(t)
                    fout.write(str(t[0]) + '\n')

                print("pred:1", pt[0][0])
                fout.write("pred1: " + str(pt[0][0]) + '\n')
                print("pred:2", pt2[0][0])
                fout.write("pred2: " + str(pt2[0][0]) + '\n\n')

                print()

    model.dec.set_teacher_forcing(cur_tc)
    fout.close()


def chat_inference_beam(model, inv_dict, config):
    criteria = nn.CrossEntropyLoss(ignore_index=0, size_average=False)
    if use_cuda:
        criteria.cuda()

    model.dec.set_teacher_forcing(True)
    model.load_state_dict(torch.load(config.name))
    model.eval()

    with open('./data/vocab.pkl', 'rb') as f:
        _ = pickle.load(f)
        w2i = pickle.load(f)
    while True:
        s = input()
        if s == 'q':
            print('exit')
            break
        t = [1]
        for word in s:
            if word == ' ':
                t.append(w2i['^'])
            else:
                t.append(w2i[word])

        t.append(2)
        temp = torch.tensor(t, device=config.device, dtype=torch.int64)
        temp = temp.unsqueeze(0).unsqueeze(0).contiguous().clone()
        qu_seq = model.base_enc(temp.view(-1, temp.size(2))).view(temp.size(0), temp.size(1), -1)
        final_session_o = model.ses_enc(qu_seq, None)

        while True:
            sent = generate(model, final_session_o[0, :, :].unsqueeze(0), config)

            pt = tensor_to_sent(sent, inv_dict)
            print("response: ", pt[0][0])

            temp = torch.tensor(sent[0][0], device=config.device, dtype=torch.int64)
            temp = temp.unsqueeze(0).unsqueeze(0).contiguous().clone()
            qu_seq = model.base_enc(temp.view(-1, temp.size(2))).view(temp.size(0), temp.size(1), -1)
            final_session_o = model.ses_enc(qu_seq, final_session_o)

            s = input()
            if s == 'q':
                print('reset')
                break
            t = [1]
            for word in s:
                if word == ' ':
                    t.append(w2i['^'])
                else:
                    t.append(w2i[word])
            t.append(2)
            temp = torch.tensor(t, device=config.device, dtype=torch.int64)
            temp = temp.unsqueeze(0).unsqueeze(0).contiguous().clone()
            qu_seq = model.base_enc(temp.view(-1, temp.size(2))).view(temp.size(0), temp.size(1), -1)
            final_session_o = model.ses_enc(qu_seq, final_session_o)

