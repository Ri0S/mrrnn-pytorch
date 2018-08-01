import torch
import pickle

from configs import config
from torch.utils.data import Dataset


def custom_collate_fn(batch):
    char = torch.tensor(batch[0][0], dtype=torch.int64, device=config.device).squeeze(0)
    charLength = torch.tensor(batch[0][1], dtype=torch.int64, device=config.device).squeeze(0)
    word = torch.tensor(batch[0][2], dtype=torch.int64, device=config.device).squeeze(0)
    wordLength = torch.tensor(batch[0][3], dtype=torch.int64, device=config.device).squeeze(0)

    return char.transpose(0, 1), charLength.transpose(0, 1), word.transpose(0, 1), wordLength.transpose(0, 1)


class MovieTriples(Dataset):
    def __init__(self, data_type, batch_size):
        char_path = './data/' + data_type + '_char.pkl'
        word_path = './data/' + data_type + '_word.pkl'
        char_length_path = './data/' + data_type + '_char_length.pkl'
        word_length_path = './data/' + data_type + '_word_length.pkl'
        length_path = './data/' + data_type + '_length.txt'


        self.cds = []
        self.wds = []
        self.clength = []
        self.wlength = []

        with open(char_path, 'rb') as f, open(word_path, 'rb') as f2, \
                open(char_length_path, 'rb') as f3, open(word_length_path, 'rb') as f4:
            char = pickle.load(f)
            word = pickle.load(f2)
            charLength = pickle.load(f3)
            wordLength = pickle.load(f4)

            index = 0
            with open(length_path, encoding='utf-8') as f3:
                length = f3.read().split()
                for tl, seqlen in enumerate(length):
                    word_data = []
                    char_data = []
                    chardLength = []
                    worddLength = []

                    char_maxlen = 0
                    word_maxlen = 0

                    for idx in range(int(seqlen)):
                        char_us = char[index]
                        word_us = word[index]
                        charUsLength = charLength[index]
                        wordUsLength = wordLength[index]
                        index += 1

                        if char_maxlen < max([len(a) for a in char_us]):
                            char_maxlen = max([len(a) for a in char_us])
                        if word_maxlen < max([len(a) for a in word_us]):
                            word_maxlen = max([len(a) for a in word_us])

                        char_data.append(char_us)
                        word_data.append(word_us)
                        chardLength.append(charUsLength)
                        worddLength.append(wordUsLength)

                        if idx % batch_size == batch_size - 1:
                            for i in range(batch_size):
                                for j in range(tl + 2):
                                    char_data[i][j].extend([0 for _ in range(char_maxlen - len(char_data[i][j]))])
                                    word_data[i][j].extend([0 for _ in range(word_maxlen - len(word_data[i][j]))])
                            self.cds.append(char_data)
                            self.wds.append(word_data)
                            self.clength.append(chardLength)
                            self.wlength.append(worddLength)
                            char_maxlen = 0
                            word_maxlen = 0

                            char_data = []
                            word_data = []
                            chardLength = []
                            worddLength = []

                    if len(char_data) != 0:
                        for i in range(len(char_data)):
                            for j in range(tl + 2):
                                char_data[i][j].extend([0 for _ in range(char_maxlen - len(char_data[i][j]))])
                                word_data[i][j].extend([0 for _ in range(word_maxlen - len(char_data[i][j]))])
                        self.cds.append(char_data)
                        self.wds.append(word_data)
                        self.clength.append(chardLength)
                        self.wlength.append(worddLength)
                    # break  # single turn test

    def __len__(self):
        return len(self.cds)

    def __getitem__(self, idx):
        return self.cds[idx], self.clength[idx], self.wds[idx], self.wlength[idx]


MovieTriples('train', 32)