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


```
pip install -r requirements.txt
```

## Reference
- 딥러닝을 이용한 자연어 처리 입문 (https://wikidocs.net/31379)
- 김기현의 BERT, GPT-3를 활용한 자연어처리 올인원 패키지 Online
- 허깅페이스 공식 문서 (https://huggingface.co/docs/transformers/v4.22.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)

