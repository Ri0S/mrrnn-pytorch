from modules import *


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.base_enc = BaseEncoder(config.syllable_size, config.embedding_size, config.encoder_hidden_size)
        self.ses_enc = SessionEncoder(config.context_hidden_size, config.encoder_hidden_size)
        self.dec = Decoder()
        if config.model.lower() == 'mrrnn':
            self.coarse = Coarse()

    def forward(self, syllable_data, syllable_length, coarse_data, coarse_length):
        us = syllable_data[:-1].transpose(0, 1).contiguous().clone()
        us_length = syllable_length[:-1].transpose(0, 1).contiguous().clone()
        gold = syllable_data[-1]
        gold_length = syllable_length[-1]
        qu_seq = self.base_enc(us.view(-1, us.size(2)), us_length.view(-1)).view(us.size(0), us.size(1), -1)
        final_session_o = self.ses_enc(qu_seq, None)
        if config.model.lower() == 'mrrnn':
            coarse_hidden, loss = self.coarse(coarse_data, coarse_length)
            final_session_o = torch.cat((final_session_o, coarse_hidden), 2)

        preds = self.dec((final_session_o, gold), gold_length)

        if config.model.lower() == 'mrrnn':
            return preds, loss

        return preds


class Coarse(nn.Module):
    def __init__(self):
        super(Coarse, self).__init__()
        self.base_enc = BaseEncoder(config.word_size, config.embedding_size, config.encoder_hidden_size)
        self.ses_enc = SessionEncoder(config.context_hidden_size, config.encoder_hidden_size, 'coarse')
        self.dec = Decoder('coarse')
        self.coarse_enc = CoarseEncoder(config.word_size, config.embedding_size, config.encoder_hidden_size)
        self.criteria = nn.CrossEntropyLoss(ignore_index=0, size_average=False)

    def forward(self, coarse_data, coarse_length):
        us = coarse_data[:-1].transpose(0, 1).contiguous().clone()
        us_length = coarse_length[:-1].transpose(0, 1).contiguous().clone()
        qu_seq = self.base_enc(us.view(-1, us.size(2)), us_length.view(-1)).view(us.size(0), us.size(1), -1)
        final_session_o = self.ses_enc(qu_seq, None)
        loss = 0
        pred_words = [torch.tensor([], device=config.device, dtype=torch.int64) for _ in range(coarse_data.size(1))]
        if config.mode == 'train':
            for i in range(final_session_o.size(1)):  # sequence
                preds, words = self.dec((final_session_o[:, i:i + 1, :], coarse_data[i + 1, :, :]),
                                        coarse_length[i + 1, :])
                loss += self.criteria(preds.view(-1, config.word_size), coarse_data[i + 1].contiguous().view(-1))
                # words = [words[idx][:coarse_length[i + 1][idx]] for idx in range(coarse_data.size(1))]  # batch
                words = [coarse_data[i+1][idx][:coarse_length[i + 1][idx]] for idx in range(coarse_data.size(1))]  # batch teacher forcing?
                pred_words = [torch.cat((pred_words[idx], words[idx])) for idx in range(coarse_data.size(1))]

            coarse_hidden = self.coarse_enc(pred_words, coarse_length[1:].sum(0))
            total_length = coarse_length[1:].sum()
        elif config.mode == 'test':
            i = 0
            for i in range(final_session_o.size(1) - 1):
                words = [coarse_data[i + 1][idx][:coarse_length[i + 1][idx]] for idx in
                         range(coarse_data.size(1))]  # batch teacher forcing?
                pred_words = [torch.cat((pred_words[idx], words[idx])) for idx in range(coarse_data.size(1))]
            words = self.dec((final_session_o[:, -1, :].unsqueeze(1), coarse_data[-1, :, :], None),
                             coarse_length[i + 1, :])
            words = [words[i][:21 if len(words.eq(3)[i].ne(0).nonzero()) == 0 else
            words.eq(3)[i].ne(0).nonzero()[0].item() + 1] for i in range(words.size(0))]
            pred_length = torch.tensor([len(word) for word in words], dtype=torch.int64, device=config.device)
            pred_words = [torch.cat((pred_words[idx], words[idx])) for idx in range(coarse_data.size(1))]

            coarse_hidden = self.coarse_enc(pred_words, pred_length if len(coarse_length[1:-1]) == 0 else coarse_length[1:-1].sum(0) + pred_length)
            total_length = (coarse_length[1:-1].sum(0) + pred_length).sum()
        return coarse_hidden, loss / total_length.item()

