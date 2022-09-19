"""
BERT next sentence prediction (NSP) example
"""
import tensorflow as tf
from transformers import TFBertForNextSentencePrediction
from transformers import AutoTokenizer


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
