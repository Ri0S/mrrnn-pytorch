import torch
import torch.nn.init as init


def tensor_to_sent(x, inv_dict, greedy=False):
    sents = []
    inv_dict[0] = '<pad>'
    for li in x:
        if not greedy:
            scr = li[1]
            seq = li[0]
        else:
            scr = 0
            seq = li
        sent = []
        for i in seq:
            sent.append(inv_dict[i])
            if i == 2:
                break
        sents.append((" ".join(sent), scr))
    return sents


def get_sent_ll(u3, u3_lens, model, criteria, ses_encoding):
    preds, _ = model.dec([ses_encoding, u3, u3_lens])
    preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
    u3 = u3[:, 1:].contiguous().view(-1)
    loss = criteria(preds, u3).data[0]
    target_toks = u3.ne(0).long().sum().data[0]
    return -1 * loss / target_toks


def load_model_state(mdl, fl):
    saved_state = torch.load(fl)
    mdl.load_state_dict(saved_state)


def sort_key(temp, mmi):
    if mmi:
        lambda_param = 0.25
        return temp[1] - lambda_param * temp[2] + len(temp[0]) * 0.1
    else:
        return temp[1] / len(temp[0]) ** 0.7


def init_param(model):
    for name, param in model.named_parameters():
        if 'embed' in name:
            continue
        elif ('rnn' in name or 'lm' in name) and len(param.size()) >= 2:
            init.orthogonal(param)
        else:
            init.normal(param, 0, 0.01)


def clip_gnorm(model):
    for name, p in model.named_parameters():
        param_norm = p.grad.data.norm()
        if param_norm > 1:
            p.grad.data.mul_(1 / param_norm)
