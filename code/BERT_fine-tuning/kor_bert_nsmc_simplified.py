# -*- coding: utf-8 -*-
"""
네이버 영화 리뷰 평가 예측 => 라이브러리 적극 활용

BERT + Dense layer => TFBertSequenceClassification 대체
"""
import pandas as pd
import urllib.request


urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')
train_data.drop_duplicates(subset=['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
# train_data['label'].value_counts().plot(kind = 'bar')
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거

"""# 토크나이저를 이용한 정수 인코딩
이미 학습해놓은 모델을 사용한다고 하면  
1. 토크나이저 (이 모델이 만들어졌을 당시에 '사과' 라는 단어가 36번이었다. 정보를 알기 위해)  
2. 모델  
이 두 가지를 로드해야 합니다.
"""
import transformers
from transformers import BertTokenizerFast


tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
test_data = test_data.dropna(how='any')

X_train_list = train_data['document'].tolist()
X_test_list = test_data['document'].tolist()
y_train = train_data['label'].tolist()
y_test = test_data['label'].tolist()

X_train = tokenizer(X_train_list, truncation=True, padding=True)
X_test = tokenizer(X_test_list, truncation=True, padding=True)

print(X_train[0].tokens)
print(X_train[0].ids)
print(X_train[0].type_ids)  # segment idx
print(X_train[0].attention_mask)

"""# 데이터셋 생성 및 모델 학습"""

import tensorflow as tf

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(X_train),  # token_ids, segment_idx, attention_mask which are inputs for BERT
    y_train
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(X_test),
    y_test
))

from transformers import TFBertForSequenceClassification
from tensorflow.keras.callbacks import EarlyStopping

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

model = TFBertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2, from_pt=True)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

callback_earlystop = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.001,
    patience=2)

model.fit(
    train_dataset.shuffle(10000).batch(32), epochs=5, batch_size=64,
    validation_data=val_dataset.shuffle(10000).batch(64),
    callbacks=[callback_earlystop]
)

model.evaluate(val_dataset.batch(1024))

"""# 모델 저장"""

model.save_pretrained('nsmc_model/bert-base')
tokenizer.save_pretrained('nsmc_model/bert-base')

"""# 모델 로드 및 테스트"""

from transformers import TextClassificationPipeline

# 로드하기
loaded_tokenizer = BertTokenizerFast.from_pretrained('nsmc_model/bert-base')
loaded_model = TFBertForSequenceClassification.from_pretrained('nsmc_model/bert-base')

text_classifier = TextClassificationPipeline(
    tokenizer=loaded_tokenizer,
    model=loaded_model,
    framework='tf',
    return_all_scores=True
)


print(text_classifier('뭐야 이 평점들은.... 나쁘진 않지만 10점 짜리는 더더욱 아니잖아')[0])
print(text_classifier('오랜만에 평점 로긴했네ㅋㅋ 킹왕짱 쌈뽕한 영화를 만났습니다 강렬하게 육쾌함')[0])


