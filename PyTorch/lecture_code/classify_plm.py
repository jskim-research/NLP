import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)  # 예측 class 들 중 확률 높은 k개
    p.add_argument('--data_fn', required=True)

    config = p.parse_args()

    return config


def read_text(fn):
    '''
    Read text from standard input for inference.
    '''
    with open(fn, 'r', encoding="UTF-8") as f:
        lines = f.readlines()

        labels, texts = [], []
        for line in lines:
            if line.strip() != '':
                # The file should have tab delimited two columns.
                # First column indicates label field,
                # and second column indicates text field.
                label, text = line.strip().split('\t')
                labels += [label]
                texts += [text]
    return texts


def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
        # load 할 때 cpu 또는 원하는 gpu에 바로 매핑
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']

    lines = read_text(config.data_fn)  # 대용량 데이터의 경우 밑 추론 for 문에서 generator로 가져오는 게 좋긴 함

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        # dictionary에서 가져와도 되긴 하는데.
        tokenizer = BertTokenizerFast.from_pretrained(train_config.pretrained_model_name)
        model_loader = AlbertForSequenceClassification if train_config.use_albert else BertForSequenceClassification
        model = model_loader.from_pretrained(
            train_config.pretrained_model_name,
            num_labels=len(index_to_label)
        )
        model.load_state_dict(bert_best)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        # model에서 바로 device를 알 수 있는게 아니라 parameter가 매핑된 device를 확인
        # 변수들도 똑같은 device에 매핑하는 것이 좋을테니까.
        device = next(model.parameters()).device

        # Don't forget turn-on evaluation mode.
        model.eval()

        y_hats = []
        for idx in range(0, len(lines), config.batch_size):
            mini_batch = tokenizer(
                lines[idx:idx + config.batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            x = mini_batch['input_ids']
            x = x.to(device)
            # padding 된 곳 mask
            mask = mini_batch['attention_mask']
            mask = mask.to(device)

            # Take feed-forward
            # model output shape will be (batch_num, num_classes)
            # model 끝에 fc layer 붙은 것이므로 => 아니었으면 (batch_num, length, Optional[dim]) 일 것
            y_hat = F.softmax(model(x, attention_mask=mask).logits, dim=-1)

            y_hats += [y_hat]
        # Concatenate the mini-batch wise result
        y_hats = torch.cat(y_hats, dim=0)
        # |y_hats| = (len(lines), n_classes)

        probs, indice = y_hats.cpu().topk(config.top_k)
        # |indice| = (len(lines), top_k)

        for i in range(len(lines)):
            sys.stdout.write('%s\t%s\n' % (
                ' '.join([index_to_label[int(indice[i][j])] for j in range(config.top_k)]), 
                lines[i]
            ))


if __name__ == '__main__':
    config = define_argparser()
    main(config)
