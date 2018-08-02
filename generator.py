import numpy as np
import copy
import torch.functional as F

from modules import *
from util import *
from dataloader import *
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
np.random.seed(123)
if use_cuda:
    torch.cuda.manual_seed(123)


def inference(dataloader, model):
    criteria = nn.CrossEntropyLoss(ignore_index=0, size_average=False)
    if use_cuda:
        criteria.cuda()

    cur_tc = model.dec.get_teacher_forcing()
    model.dec.set_teacher_forcing(True)
    fout = open(config.saved_model + "_beam" + str(config.beam_size) + "_result.txt", 'w', encoding='utf-8')
    model.load_state_dict(torch.load('./model/' + config.saved_model))

    model.eval()

    for i_batch, (syllableData, syllableLength, coarseData, coarseLength) in enumerate(tqdm(dataloader)):
        gold = syllableData[-1]
        if config.model == 'mrrnn':
            preds, coarse_loss = model(syllableData, syllableLength, coarseData, coarseLength)
        else:
            preds = model(syllableData, syllableLength, coarseData, coarseLength)

        pred_words = preds.max(2)[1]
        pred_words = [pred_words[i][:pred_words.size(1) if len(pred_words.eq(3)[i].ne(0).nonzero()) == 0 else
        pred_words.eq(3)[i].ne(0).nonzero()[0].item() + 1] for i in range(pred_words.size(0))]

        gold = gold[:, 1:].contiguous().view(-1)

        inps = syllableData.transpose(0, 1)
        inps_length = syllableLength.transpose(0, 1)
        for i in range(inps.size(0)):  # batch
            for j in range(inps.size(1) - 1):  # sequence length
                inp = ''.join([config.i2c[a.item()] for a in inps[i][j][0:inps_length[i][j]]][1:-1]).replace('^', ' ') + '\n'
                print(inp, end='')
                fout.write(inp)
            inp = 'gold: '.join([config.i2c[a.item()] for a in inps[i][inps.size(1) - 1][0:inps_length[i][inps.size(1) - 1]]][1:-1]).replace('^', ' ') + '\n'
            print(inp, end='')
            fout.write(inp)
            pred = 'pred: '.join([config.i2c[a.item()] for a in pred_words[i]].replace('^', ' ') + '\n\n')
            print(pred, end='')
            fout.write(pred)

    model.dec.set_teacher_forcing(cur_tc)
    fout.close()

