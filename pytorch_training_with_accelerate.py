from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

# Using pytorch to write a training loop instead of using the huggingface trainer API

# This is one of the 10 datasets composing the GLUE benchmark, which is an academic benchmark that is used to measure the performance of ML models across 10 different text classification tasks.
# The dataset consists of 5,801 pairs of sentences, with a label indicating if they are paraphrases or not (i.e., if both sentences mean the same thing).
raw_datasets = load_dataset('glue', 'mrpc')

# Select the checkpoint
checkpoint = 'bert-base-uncased'
# Initialie the tokenizer using the autotokenizer and passing the checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# Note that we’ve left the padding argument out in our tokenization function for now because it will be handled by our batching.
# This is because padding all the samples to the maximum length is not efficient: it’s better to pad the samples when we’re building a batch, as then we only need to pad to the maximum length in that batch, and not the maximum length in the entire dataset.
# This can save a lot of time and processing power when the inputs have variable lengths!
# The tokenizer takes both sequences and concats them under the input_ids key
def tokenize_function(example):
  return tokenizer(example['sentence1'], example['sentence2'], truncation=True)

# Batches the dataset into smaller segments, then the whole patch is sent to the tokenize function with each column being a list of values with the key being the column name.
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Need to manually remove and edit these fields to prepare them for the model
# Models generally only accept numbers
tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
# set the format for pytorch
tokenized_datasets.set_format('torch')
print(tokenized_datasets['train'].column_names)
# ['attention_mask', 'input_ids', 'labels', 'token_type_ids']

# There are 2 sequences stored in the input_ids at this point.
print(tokenized_datasets['train'][0])
# {
#   'labels': tensor(1),
#   'input_ids': tensor([  101,  2572,  3217,  5831,  5496,  2010,  2567,  1010,  3183,  2002,
#          2170,  1000,  1996,  7409,  1000,  1010,  1997,  9969,  4487, 23809,
#          3436,  2010,  3350,  1012,   102,  7727,  2000,  2032,  2004,  2069,
#          1000,  1996,  7409,  1000,  1010,  2572,  3217,  5831,  5496,  2010,
#          2567,  1997,  9969,  4487, 23809,  3436,  2010,  3350,  1012,   102]),
#   'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1]),
#   'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1])
# }

# Dynamically pads all examples in the batch to the length of the longest one.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
from torch.utils.data import DataLoader

# shuffle=True
# Randomly shuffles the data at the start of each epoch.
# Helps reduce overfitting and ensures the model doesn’t memorize batch order.
train_dataloader = DataLoader(
    tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets['validation'], batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})
# {
#   'labels': torch.Size([8]),
#   'input_ids': torch.Size([8, 66]),
#   'token_type_ids': torch.Size([8, 66]),
#   'attention_mask': torch.Size([8, 66])
# }

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Run the batch through the model
# All transformer models will return the loss when the labels are provided
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
# tensor(0.7694, grad_fn=<NllLossBackward0>) torch.Size([8, 2])


# The same optimizer used by the Trainer module
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
optimizer = AdamW(model.parameters(), lr=5e-5)


accelerator = Accelerator()

# Accelerate handles the device placement for you
# This will wrap those objects in the proper container to make sure your distributed training works as intended.
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
  'linear',
  optimizer=optimizer,
  num_warmup_steps=0,
  num_training_steps=num_training_steps,
)
print(num_training_steps)
# 1377

import torch
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
  for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)
    