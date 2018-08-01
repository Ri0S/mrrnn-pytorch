import argparse
import pickle
import pprint
import torch

import torch.nn as nn

from pathlib import Path
from torch import optim


project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir.joinpath('datasets')
data_dict = {'cornell': data_dir.joinpath('cornell'), 'ubuntu': data_dir.joinpath('ubuntu')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
rnn_dict = {'lstm': nn.LSTM, 'gru': nn.GRU}
username = Path.home().name
save_dir = Path(f'/data1/{username}/conversation/')


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""

        with open('./data/i2c.pkl', 'rb') as f:
            self.i2c = pickle.load(f)
        with open('./data/c2i.pkl', 'rb') as f:
            self.c2i = pickle.load(f)
        with open('./data/i2w.pkl', 'rb') as f:
            self.i2w = pickle.load(f)
        with open('./data/w2i.pkl', 'rb') as f:
            self.w2i = pickle.load(f)

        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        # Pickled Vocabulary
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.syllable_size = len(self.i2c)
        self.word_size = len(self.i2w)

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model', type=str, default='mrrnn')
    # Train
    parser.add_argument('--succeed', type=str2bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default=None)

    # Generation
    parser.add_argument('--beam_size', type=int, default=1)

    # Model
    parser.add_argument('--saved_model', type=str, default='000')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', type=str2bool, default=True)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--encoder_hidden_size', type=int, default=400)
    parser.add_argument('--context_hidden_size', type=int, default=400)
    parser.add_argument('--decoder_hidden_size', type=int, default=800)
    parser.add_argument('--dropout', type=float, default=0.4)

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


config = get_config()