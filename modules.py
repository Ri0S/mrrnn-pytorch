import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from configs import config

use_cuda = torch.cuda.is_available()


def max_out(x):
    if len(x.size()) == 2:
        s1, s2 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2 // 2, 2)
        x, _ = torch.max(x, 2)

    elif len(x.size()) == 3:
        s1, s2, s3 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2, s3 // 2, 2)
        x, _ = torch.max(x, 3)

    return x


class BaseEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size):
        super(BaseEncoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = config.num_layers
        self.drop = nn.Dropout(config.dropout)
        self.direction = 2 if config.bidirectional else 1
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=0, sparse=False)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hid_size,
                          num_layers=self.num_lyr, bidirectional=config.bidirectional, batch_first=True, dropout=config.dropout)

    def forward(self, inp, length):
        length, indices = length.sort(descending=True)
        x = inp.index_select(0, indices)

        bt_siz, seq_len = x.size(0), x.size(1)

        h_0 = torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size, device=config.device, dtype=torch.float32)
        if use_cuda:
            h_0 = h_0.cuda()
        x_emb = self.embed(x)
        x_emb = self.drop(x_emb)
        x_emb = pack_padded_sequence(x_emb, length, True)

        x_o, x_hid = self.rnn(x_emb, h_0)
        if self.direction == 2:
            x_hids = []
            for i in range(self.num_lyr):
                x_hid_temp, _ = torch.max(x_hid[2*i:2*i + 2, :, :], 0, keepdim=True)
                x_hids.append(x_hid_temp)
            x_hid = torch.cat(x_hids, 0)
        
        x_hid = x_hid[self.num_lyr-1, :, :].unsqueeze(0)
        x_hid = x_hid.transpose(0, 1)
        _, reverse_indices = indices.sort()
        x_hid = x_hid.index_select(0, reverse_indices)
        return x_hid


class CoarseEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size):
        super(CoarseEncoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = config.num_layers
        self.drop = nn.Dropout(config.dropout)
        self.direction = 2 if config.bidirectional else 1
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=0, sparse=False)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hid_size,
                          num_layers=self.num_lyr, bidirectional=config.bidirectional, batch_first=True,
                          dropout=config.dropout)

    def forward(self, inp, length):
        length, indices = length.sort(descending=True)
        x = pad_sequence([inp[idx] for idx in indices], True, 0)

        bt_siz, seq_len = x.size(0), x.size(1)

        h_0 = torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size, device=config.device, dtype=torch.float32)
        if use_cuda:
            h_0 = h_0.cuda()
        x_emb = self.embed(x)
        x_emb = self.drop(x_emb)
        x_emb = pack_padded_sequence(x_emb, length, True)

        x_o, x_hid = self.rnn(x_emb, h_0)
        if self.direction == 2:
            x_hids = []
            for i in range(self.num_lyr):
                x_hid_temp, _ = torch.max(x_hid[2 * i:2 * i + 2, :, :], 0, keepdim=True)
                x_hids.append(x_hid_temp)
            x_hid = torch.cat(x_hids, 0)

        x_hid = x_hid[self.num_lyr - 1, :, :].unsqueeze(0)
        x_hid = x_hid.transpose(0, 1)
        _, reverse_indices = indices.sort()
        x_hid = x_hid.index_select(0, reverse_indices)
        return x_hid


class SessionEncoder(nn.Module):
    def __init__(self, hid_size, inp_size, mode=None):
        super(SessionEncoder, self).__init__()
        self.mode = mode
        self.hid_size = hid_size
        self.rnn = nn.GRU(hidden_size=hid_size, input_size=inp_size,
                          num_layers=1, bidirectional=False, batch_first=True, dropout=config.dropout)

    def forward(self, x, h_0):
        if h_0 is None:
            h_0 = torch.zeros(1, x.size(0), self.hid_size, device=config.device, dtype=torch.float32)

        if use_cuda:
            h_0 = h_0.cuda()
        h_o, h_n = self.rnn(x, h_0)
        h_n = h_n.view(x.size(0), -1, self.hid_size)
        if self.mode == 'coarse':
            return h_o
        return h_n


class Decoder(nn.Module):
    def __init__(self, mode=None):
        super(Decoder, self).__init__()
        self.mode = mode
        self.emb_size = config.embedding_size
        self.hid_size = config.decoder_hidden_size
        self.num_lyr = 1
        self.teacher_forcing = True

        self.drop = nn.Dropout(config.dropout)
        self.tanh = nn.Tanh()

        if mode == 'coarse':
            self.embed_in = nn.Embedding(config.word_size, self.emb_size, padding_idx=0, sparse=False)
            self.embed_out = nn.Linear(self.emb_size, config.word_size, bias=False)
        else:
            self.embed_in = nn.Embedding(config.syllable_size, self.emb_size, padding_idx=0, sparse=False)
            self.embed_out = nn.Linear(self.emb_size, config.syllable_size, bias=False)
        
        self.rnn = nn.GRU(hidden_size=self.hid_size, input_size=self.emb_size,num_layers=self.num_lyr, batch_first=True, dropout=config.dropout)

        if config.model == 'mrrnn' and mode is None:
            self.ses_to_dec = nn.Linear(config.context_hidden_size * 2, self.hid_size)
            self.ses_inf = nn.Linear(config.context_hidden_size * 2, self.emb_size * 2, False)
        else:
            self.ses_to_dec = nn.Linear(config.context_hidden_size, self.hid_size)
            self.ses_inf = nn.Linear(config.context_hidden_size, self.emb_size * 2, False)
        self.dec_inf = nn.Linear(self.hid_size, self.emb_size*2, False)
        self.emb_inf = nn.Linear(self.emb_size, self.emb_size*2, True)
        self.tc_ratio = 0.2

    def do_decode_tc(self, ses_encoding, target, length):
        length, indices = length.sort(descending=True)
        x = target.index_select(0, indices)
        x = target

        target_emb = self.embed_in(x)
        target_emb = self.drop(target_emb)
        emb_inf_vec = self.emb_inf(target_emb)
        target_emb = pack_padded_sequence(target_emb, length, True)

        init_hidn = self.tanh(self.ses_to_dec(ses_encoding))
        init_hidn = init_hidn.view(self.num_lyr, target.size(0), self.hid_size)
        
        hid_o, _ = self.rnn(target_emb, init_hidn)
        hid_o = pad_packed_sequence(hid_o, True, 0, target.size(1))

        _, reverse_indices = indices.sort()
        hid_o = hid_o[0].index_select(0, reverse_indices)

        
        dec_hid_vec = self.dec_inf(hid_o)
        ses_inf_vec = self.ses_inf(ses_encoding)
        total_hid_o = dec_hid_vec + ses_inf_vec + emb_inf_vec
        
        hid_o_mx = max_out(total_hid_o)
        hid_o_mx = self.embed_out(hid_o_mx)

        if self.mode == 'coarse':
            return hid_o_mx, hid_o_mx.max(2)[1]
        else:
            return hid_o_mx
        
    def do_decode(self, siz, seq_len, ses_encoding, target, length):
        ses_inf_vec = self.ses_inf(ses_encoding)
        ses_encoding = self.tanh(self.ses_to_dec(ses_encoding))
        hid_n, preds, lm_preds = ses_encoding, [], []
        
        hid_n = hid_n.view(self.num_lyr, siz, self.hid_size)
        inp_tok = torch.ones(siz, 1, device=config.device, dtype=torch.int64).new_full((siz, 1), 2)
        out_words = torch.ones(siz, 1, device=config.device, dtype=torch.int64).new_full((siz, 1), 2)
        for i in range(seq_len):
            inp_tok_vec = self.embed_in(inp_tok)
            emb_inf_vec = self.emb_inf(inp_tok_vec)
            
            inp_tok_vec = self.drop(inp_tok_vec)
            
            hid_o, hid_n = self.rnn(inp_tok_vec, hid_n)
            dec_hid_vec = self.dec_inf(hid_o)
            
            total_hid_o = dec_hid_vec + ses_inf_vec + emb_inf_vec
            hid_o_mx = max_out(total_hid_o)

            hid_o_mx = self.embed_out(hid_o_mx)
            # preds.append(hid_o_mx)
            _, inp_tok = torch.max(hid_o_mx, dim=2)
            out_words = torch.cat((out_words, inp_tok), 1)

        # dec_o = torch.cat(preds, 1)
        # return dec_o
        return out_words

    def forward(self, input, length):
        if len(input) == 1:
            ses_encoding = input
            x = None
        elif len(input) == 2:
            ses_encoding, x = input
        else:
            ses_encoding, x, beam = input

        siz, seq_len = x.size(0), x.size(1)

        if len(input) == 3 and beam is None:
            dec_o = self.do_decode(siz, 20, ses_encoding, None, None)

        elif self.teacher_forcing:
            dec_o = self.do_decode_tc(ses_encoding, x, length)
        else:
            dec_o = self.do_decode(siz, seq_len, ses_encoding, x, length)
            
        return dec_o

    def set_teacher_forcing(self, val):
        self.teacher_forcing = val
    
    def get_teacher_forcing(self):
        return self.teacher_forcing
    
    def set_tc_ratio(self, new_val):
        self.tc_ratio = new_val
    
    def get_tc_ratio(self):
        return self.tc_ratio

