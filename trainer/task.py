from datasets import load_dataset
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch import optim
from torch.utils.data import Dataset
from tqdm import tqdm


def get_dataloader(dataset, tokenizer, batch_size=32):
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length'), batched=True)
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


if __name__ == '__main__':
    MODEL_NAME = 'roberta-large'
    EPOCHS = 30
    BATCH_SIZE = 32

    # load, encode and format input data for pytorch model
    datasets = load_dataset('ag_news')
    train_ds = datasets['train']
    test_ds = datasets['test']
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    train_dataloader = get_dataloader(train_ds, tokenizer, batch_size=BATCH_SIZE)
    test_dataloader = get_dataloader(test_ds, tokenizer, batch_size=BATCH_SIZE)

    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME)

    # fine tune the model on the new dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    for epoch in range(EPOCHS):
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 10 == 0:
                print(f"loss: {loss}")
