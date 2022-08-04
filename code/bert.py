import pandas as pd
import tensorflow as tf
from transformers import FillMaskPipeline
from transformers import BertTokenizer
from transformers import TFBertForNextSentencePrediction
from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer


# BERT => subword tokenizer WordPiece 사용  (BPE와 유사한 방식)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Bert-base의 토크나이저

result = tokenizer.tokenize('Here is the sentence I want embeddings for.')
print(result)
print(tokenizer.vocab['here'])  # char to int

# BERT의 단어 집합을 vocabulary.txt에 저장
with open('vocabulary.txt', 'w', encoding="utf-8") as f:
    for token in tokenizer.vocab.keys():
        f.write(token + '\n')

df = pd.read_fwf('vocabulary.txt', header=None)
print(df)
print('단어 집합의 크기 :', len(df))
# BERT 특별 토큰: [PAD], [UNK], [CLS], [SEP], [MASK] => 0, 100, 101, 102, 103
print(df.loc[102].values[0])

# BERT 학습 task
# 1. Masked Language Model, MLM => 전체 token 15% => 1.1 [MASKED] 치환 (80%) 1.2 RANDOM token 치환 1.3 그대로 놔둠
# 무슨 짓을 했든 간에 원래 token을 알아내야함 (1.2 => 이게 이 자리 맞나? 1.3 => 무조건 바꾸는건 안돼..)
# 2. Next Sentence Prediction, NSP => 문장 간의 관계 파악

# BERT Embedding
# position embedding + WordPiece embedding + Segment embedding

# BERT 학습 관련 사항
# 훈련 데이터는 위키피디아(25억 단어)와 BooksCorpus(8억 단어) ≈ 33억 단어
# WordPiece 토크나이저로 토큰화를 수행 후 15% 비율에 대해서 마스크드 언어 모델 학습
# 두 문장 Sentence A와 B의 합한 길이. 즉, 최대 입력의 길이는 512로 제한
# 100만 step 훈련 ≈ (총 합 33억 단어 코퍼스에 대해 40 에포크 학습)
# 옵티마이저 : 아담(Adam)
# 학습률(learning rate) : 10e-4
# 가중치 감소(Weight Decay) : L2 정규화로 0.01 적용
# 드롭 아웃 : 모든 레이어에 대해서 0.1 적용
# 활성화 함수 : relu 함수가 아닌 gelu 함수
# 배치 크기(Batch size) : 256

# BERT attention mask => padding 등은 0으로 제외

# 학습된 모델과 학습 당시 사용된 tokenizer 불러옴
model = TFBertForMaskedLM.from_pretrained('bert-large-uncased')
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
inputs = tokenizer('Soccer is a really fun [MASK].', return_tensors='tf')

# input_ids => encoded sequence
# token_type_ids => segment #
# attention mask => ignore padding token (zero)
for inp in inputs:
    print(inp)
    print(inputs[inp])

pip = FillMaskPipeline(model=model, tokenizer=tokenizer)
print(pip('Soccer is a really fun [MASK].'))
print(pip('The Avengers is a really fun [MASK].'))

model = TFBertForNextSentencePrediction.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "pizza is eaten with the use of a knife and fork. In casual settings, however, it is cut into wedges to be eaten while held in the hand."
encoding = tokenizer(prompt, next_sentence, return_tensors='tf')

for inp in encoding:
    print(inp)
    print(encoding[inp])

print(tokenizer.cls_token, ':', tokenizer.cls_token_id)
print(tokenizer.sep_token, ':', tokenizer.sep_token_id)
print(tokenizer.decode(encoding['input_ids'][0]))  # prompt, next_sentence encoding 결과 decode

logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]
softmax = tf.keras.layers.Softmax()
probs = softmax(logits)
print(probs)
print('최종 예측 레이블 :', tf.math.argmax(probs, axis=-1).numpy())

