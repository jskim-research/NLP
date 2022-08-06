"""
BERT masked language model example
"""
from transformers import FillMaskPipeline
from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer


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