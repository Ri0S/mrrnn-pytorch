# mrrnn-pytorch
pytorch0.4 Implementation (maybe a little different) of the paper [Multiresolution Recurrent Neural Networks: An Application to Dialogue Response Generation](https://arxiv.org/abs/1606.00776)

## Data Requirement
```
w2i.pkl         : coarse word to index dictionary pickle file   
                  eg) {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>':3, '<no_entity>': 4, ... }
i2w.pkl         : index to coarse word list pickle file   
                  eg) ['<pad'>, '<unk>', '<sos>', '<eos>', '<no_entity>', ...]
c2i.pkl         : char(or syllable or word) to index dictionary pickle file
i2c.pkl         : index to word(or syllable or char) list pickle file

input.pkl       : indexed sentences    
                  eg) [[[index of <sos>, index of dialog_0_sentence_0_word0, index of dialog_0_sentence_0_word1, ..., ],
                        [index of <sos>, index of dialog_0_sentence_1_word0, index of dialog_0_sentence_1_word1, ..., ]],
                       [[index of <sos>, index of dialog_1_sentence_0_word0, index of dialog_1_sentence_0_word1, ..., ],
                        [index of <sos>, index of dialog_1_sentence_1_word0, index of dialog_1_sentence_1_word1, ..., ]]]
input_length.pkl: length of sentences  
                  eg) [[length of dialog_0_sentence0, length of dialog_0_sentence1, ..., ],
                       [length of dialog_1_sentence0, length of dialog_1_sentence1, ..., ]]
input_length.txt: length of n(>=2) turns dialog   
                  eg) 10, 10, 9, 8, 7, ...,
```

## Training
python train.py --mode=train

see configs.py for detail option
