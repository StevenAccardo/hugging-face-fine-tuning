from datasets import load_dataset

# This is one of the 10 datasets composing the GLUE benchmark, which is an academic benchmark that is used to measure the performance of ML models across 10 different text classification tasks.
# The dataset consists of 5,801 pairs of sentences, with a label indicating if they are paraphrases or not (i.e., if both sentences mean the same thing).
raw_datasets = load_dataset('glue', 'mrpc')
print(raw_datasets)

# DatasetDict({
#     train: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 3668
#     })
#     validation: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 408
#     })
#     test: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 1725
#     })
# })

raw_train_dataset = raw_datasets['train']
print(raw_train_dataset[0])

# You can see both sentences are indeed paraphrases, and the label has an integer value of 1
# {
#   'sentence1': 'Amrozi accused his brother , whom he called ' the witness ' , of deliberately distorting his evidence .',
#   'sentence2': 'Referring to him as only ' the witness ' , Amrozi accused his brother of deliberately distorting his evidence .',
#   'label': 1,
#   'idx': 0
# }

# We can find the mappig of the label integers by looking at the dataset features
# we can see that label of 0 == not_equivalent and 1 == equivalent
print(raw_train_dataset.features)

# {
#   'sentence1': Value(dtype='string', id=None),
#   'sentence2': Value(dtype='string', id=None),
#   'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
#   'idx': Value(dtype='int32', id=None)
# }