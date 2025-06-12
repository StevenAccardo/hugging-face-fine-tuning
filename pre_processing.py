from transformers import AutoTokenizer

checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# How you can tokenize each sentence of the pair for a dataset
# tokenized_sentences_1 = tokenizer(raw_datasets['train']['sentence1'])
# tokenized_sentences_2 = tokenizer(raw_datasets['train']['sentence2'])

inputs = tokenizer('This is the first sentence.', 'This is the second one.')
print(inputs)


# token_type_ids indicates which input_ids belong to which sentence in the pair. O belonging to the first sentence, and 1 to the other.
# token_type_ids are not always present for each tokenizer/model. It depends how they were trained.
# { 
#   'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
#   'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
#   'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# }

print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

# ['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
