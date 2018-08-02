import train
import generator

from torch.utils.data import DataLoader
from models import *
from modules import *
from dataloader import *
from configs import config


def main():
    print('torch version {}'.format(torch.__version__))
    # print(config)

    model = Seq2Seq().to(config.device)

    if config.mode == 'train':
        model = model.train()
        train.train(model)
    elif config.mode == 'test':
        model = model.eval()
        test_dataset = MovieTriples('test', config.batch_size)

        test_dataloader = DataLoader(test_dataset, 1, shuffle=True, collate_fn=custom_collate_fn)
        generator.inference(test_dataloader, model)
        # chat_inference_beam(model, inv_dict, config)


main()

