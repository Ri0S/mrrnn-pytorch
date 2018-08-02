import math
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader
from modules import *
from util import *
from dataloader import *
from tqdm import tqdm
from configs import config

torch.manual_seed(123)
np.random.seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)


def train(model):
    model.train()
    optimizer = optim.Adam(model.parameters(), config.learning_rate)
    if config.succeed:
        model.load_state_dict(torch.load(config.saved_model))
    else:
        init_param(model)

    train_dataset, valid_dataset = MovieTriples('train', config.batch_size), MovieTriples('valid', config.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    criteria = nn.CrossEntropyLoss(ignore_index=0, size_average=False)

    best_vl_loss, patience, batch_id = 10000, 0, 0

    for i in range(config.n_epoch):
        if patience == config.patience:
            break
        tr_loss, tlm_loss, num_words = 0, 0, 0
        for i_batch, (syllableData, syllableLength, coarseData, coarseLength) in enumerate(tqdm(train_dataloader)):
            new_tc_ratio = 2100.0 / (2100.0 + math.exp(batch_id / 2100.0))
            model.dec.set_tc_ratio(new_tc_ratio)
            gold = syllableData[-1]
            if config.model == 'mrrnn':
                preds, coarse_loss = model(syllableData, syllableLength, coarseData, coarseLength)
            else:
                preds = model(syllableData, syllableLength, coarseData, coarseLength)

            preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
            gold = gold[:, 1:].contiguous().view(-1)
            loss = criteria(preds, gold)
            target_length = syllableLength[-1].sum().item()

            num_words += target_length
            tr_loss += loss.item()
            loss = loss / target_length
            if config.model == 'mrrnn':
                loss += coarse_loss

            optimizer.zero_grad()
            loss.backward()
            clip_gnorm(model)
            optimizer.step()

            batch_id += 1
        print('total loss:', tr_loss)
        print()
        if config.succeed:
            torch.save(model.state_dict(), './model/' + config.model + '_emb' + str(config.embedding_size) + '_' +
                                           'enc' + str(config.encoder_hidden_size) + '_' +
                                           'cxt' + str(config.context_hidden_size) + '_' +
                                           'dec' + str(config.decoder_hidden_size) + '_' +
                                           str(int(config.name[-3:]) + i + 1).zfill(3))
        else:
            torch.save(model.state_dict(), './model/' + config.model + '_emb' + str(config.embedding_size) + '_' +
                                           'enc' + str(config.encoder_hidden_size) + '_' +
                                           'cxt' + str(config.context_hidden_size) + '_' +
                                           'dec' + str(config.decoder_hidden_size) + '_' +
                                            str(i).zfill(3))

        vl_loss = calc_valid_loss(valid_dataloader, criteria, model)
        print("Training loss {}  Valid loss {}".format(tr_loss / num_words, vl_loss))
        if vl_loss < best_vl_loss or config.toy:
            best_vl_loss = vl_loss
            patience = 0
        else:
            patience += 1


def calc_valid_loss(data_loader, criteria, model):
    model.eval()
    cur_tc = model.dec.get_teacher_forcing()
    model.dec.set_teacher_forcing(True)
    valid_loss, num_words = 0, 0

    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        trainData = sample_batch[:-1]
        gold = sample_batch[-1]
        preds, lmpreds = model(trainData, gold)

        preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
        gold = gold[:, 1:].contiguous().view(-1)
        # do not include the lM loss, exp(loss) is perplexity
        loss = criteria(preds, gold)
        num_words += gold.ne(0).long().sum().data[0]
        valid_loss += loss.item()

    model.train()
    model.dec.set_teacher_forcing(cur_tc)

    return valid_loss / int(num_words)

