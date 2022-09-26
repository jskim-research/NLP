import torch
from torch.utils.data import Dataset


class TextClassificationCollator:
    """
    Collate function

    Dataset이 만들어낸 batch([data[random_index1], data[random_index2], ...])를 입력으로 받음.

    자연어 처리시에 mini-batch 별로 length를 맞춰줘야 하는데 이럴 때 유용
    """
    def __init__(self, tokenizer, max_length, with_text=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples):
        """
        Args:
            samples: list of Dataset items from __getitem__()
        """
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]

        encoding = self.tokenizer(
            texts,
            padding=True,  # batch 중 가장 longest에 길이를 맞춤
            truncation=True,  # max_length 이상은 자름
            return_tensors="pt",  # pytorch tensor로 반환
            max_length=self.max_length
        )

        return_value = {
            'input_ids': encoding['input_ids'],  # texts => tokens
            'attention_mask': encoding['attention_mask'],  # padding 부분 masking
            'labels': torch.tensor(labels, dtype=torch.long),
        }
        if self.with_text:
            return_value['text'] = texts

        return return_value


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        return {
            'text': text,
            'label': label,
        }


if __name__ == "__main__":
    pass

