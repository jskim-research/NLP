# Natural language processing(NLP) 프로젝트
## 개요
- Pretrained language model(PLM)을 활용한 downstream task 수행

## 개발 환경
Windows 10 + RTX 3080 + CUDA 11.1 환경에서 수행함 

Windows 10에서 torchtext >= 0.9 버전을 사용해야하므로 pytorch는 1.8.0 버전 사용 (pytorch, torchtext compatibility: https://pypi.org/project/torchtext/0.12.0/)

Lecture code에서 요구하는 버전 명시
- Python 3.6 or higher (python 3.7 사용)
- PyTorch 1.6 or higher
- PyTorch Ignite
- TorchText 0.5 or higher
- [torch-optimizer 0.0.1a15](https://pypi.org/project/torch-optimizer/)
- Tokenized corpus (e.g. [Moses](https://www.nltk.org/_modules/nltk/tokenize/moses.html), Mecab, [Jieba](https://github.com/fxsjy/jieba))
- Huggingface

```bash
$ pip install -r requirements.txt
```

## Usage

```bash
$ cd PyTorch/lecture_code
$ mkdir saved_models
$ python finetune_plm_native.py --model ./saved_models/review.native.kcbert.pth --train_fn ./data/review.sorted.uniq.refined.shuf.train.tsv --gpu_id 0 --batch_size 80 --n_epochs 2
$ cat ./data/review.sorted.uniq.refined.shuf.test.tsv | shuf | head -n 20 | awk -F'\t' ' { print $2 } ' | python classify_plm.py --model_fn ./saved_models/review.native.kcbert.pth --gpu_id 0
$ python classify_plm.py --model_fn ./saved_models/review.native.kcbert.pth --data_fn ./data/review.sorted.uniq.refined.shuf.test.tsv --gpu_id 0 | awk -F'\t' '{ print $1 }' > ./saved_models/review.native.kcbert.pth.result.txt
$ cat ./data/review.sorted.uniq.refined.shuf.test.tsv | awk -F'\t' ' { print $1 } ' > ./saved_models/test_ground_truth.txt

```

## Tips

1. 맞춘 것도, 틀린 것도 시각화 등으로 한번 봐보자

## Reference
- 딥러닝을 이용한 자연어 처리 입문 (https://wikidocs.net/31379)
- 김기현의 BERT, GPT-3를 활용한 자연어처리 올인원 패키지 Online
- 허깅페이스 공식 문서 (https://huggingface.co/docs/transformers/v4.22.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)

