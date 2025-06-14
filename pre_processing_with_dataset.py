from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("glue", "mrpc")

# One method for tokenizing the dataset and their pairs of sentences.
# It will also only work if you have enough RAM to store your whole dataset during the tokenization
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

# Note that we’ve left the padding argument out in our tokenization function for now.
# This is because padding all the samples to the maximum length is not efficient: it’s better to pad the samples when we’re building a batch, as then we only need to pad to the maximum length in that batch, and not the maximum length in the entire dataset.
# This can save a lot of time and processing power when the inputs have very variable lengths!
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# Here is how we apply the tokenization function on all our datasets at once.
# We’re using batched=True in our call to map so the function is applied to multiple elements of our dataset at once, and not on each element separately.
# This allows for faster preprocessing.
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

# You can see that the attention_masks, input_ids and token_type_ids columns have been added and are now colums for each row in the datasets.
# DatasetDict({
#     train: Dataset({
#         features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
#         num_rows: 3668
#     })
#     validation: Dataset({
#         features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
#         num_rows: 408
#     })
#     test: Dataset({
#         features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
#         num_rows: 1725
#     })
# })