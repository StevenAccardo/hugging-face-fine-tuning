from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np
import evaluate

raw_datasets = load_dataset('glue', 'mrpc')

checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Note that we’ve left the padding argument out in our tokenization function for now.
# This is because padding all the samples to the maximum length is not efficient: it’s better to pad the samples when we’re building a batch, as then we only need to pad to the maximum length in that batch, and not the maximum length in the entire dataset.
# This can save a lot of time and processing power when the inputs have very variable lengths!
def tokenize_function(example):
  return tokenizer(example['sentence1'], example['sentence2'], truncation=True)

def compute_metrics(eval_preds):
  metric = evaluate.load('glue', 'mrpc')
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)

# Here is how we apply the tokenization function on all our datasets at once.
# We’re using batched=True in our call to map so the function is applied to multiple elements of our dataset at once, and not on each element separately.
# This allows for faster preprocessing.
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# The name of the directory where the trained model will be saved is passed as an argument
# This will use the defaults which should get us in the ball park.
training_args = TrainingArguments('test-trainer', eval_strategy='epoch')

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Note that when you pass a tokenizer as the processing_class, as we did here, the default data_collator used by the Trainer will be a DataCollatorWithPadding if the processing_class is a tokenizer or feature extractor, so you can skip the line data_collator=data_collator in this call.
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
# {'eval_loss': 0.6728312969207764, 'eval_accuracy': 0.8480392156862745, 'eval_f1': 0.8959731543624161, 'eval_runtime': 1.7401, 'eval_samples_per_second': 234.465, 'eval_steps_per_second': 29.308, 'epoch': 3.0} 