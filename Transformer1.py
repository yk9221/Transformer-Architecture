if __name__ == '__main__':
    # !pip install datasets
    # !pip install evaluate
    # !pip install -U sentence-transformers

    from datasets import load_dataset, DatasetDict
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import torch
    import evaluate

    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    import joblib

    torch.cuda.empty_cache()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ################# Import additional packages you need #################
    #####################################################################################
    import numpy as np
    import pandas as pd
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer

################## HELPER CODE FOR SAVING RELEVANT FILES ##################
if __name__ == '__main__':
    def in_colab():
        try:
            import google.colab
            return True
        except ImportError:
            return False

    if in_colab():
        from google.colab import drive
        drive.mount('/content/drive')
        SAVE_PATH = '/content/drive/MyDrive'
    else:
        SAVE_PATH = '.'

"""## Part 1.a

Fine-tune [TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) on AG News and evaluate the results. You can find a tutorial for loading BERT and fine-tuning [here](https://huggingface.co/docs/transformers/training). In that tutorial, you will need to change the dataset from `"yelp_review_full"` to the correct dataset path and the model from `"bert-base-uncased"` to `"huawei-noah/TinyBERT_General_4L_312D"`. You'll also need to modify the code since AG New is a four-class classification dataset (unlike the Yelp Reviews dataset, which is a five-class classification dataset).

**TODO**
* After fine-tuning the model, save model predictions on the test set to *part1_tiny_bert_model_test_prediction.csv*. The csv file should contain "index" columns, corresponding to the unique sample index, and "pred" column, the model prediction on that sample. Your model should achieve >= 80% on the test accuracy to receive a full mark.

```
index, pred
0,model_pred_value_0
1,model_pred_value_1
2,model_pred_value_2
...
```
"""

######################## DO NOT MODIFY THE CODE ########################
if __name__ == '__main__':
    dataset = load_dataset('r-three/ag_news_subset')
    model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=4)
    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    print(dataset["train"][100])
#########################################################################

def tokenize(examples):
  concat = []
  for i in range(len(examples['title'])):
    concat.append(examples['title'][i] + '-' + examples['description'][i])

  return tokenizer(concat, truncation=True)

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
  tokenized_dataset = dataset.map(tokenize, batched=True)
  metric = evaluate.load('accuracy')

  training_args = TrainingArguments(
      output_dir='part1_tiny_bert_pred',
      eval_strategy='epoch',
      report_to='none'
  )

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_dataset['train'],
      eval_dataset=tokenized_dataset['validation'],
      compute_metrics=compute_metrics,
      data_collator=data_collator
  )

  trainer.train()

  predictions = trainer.predict(tokenized_dataset['test'].remove_columns(['label']))
  part1_tiny_bert_pred = pd.DataFrame({
      'index': dataset['test']['index'],
      'pred': np.argmax(predictions.predictions, axis=1)
  })
  part1_tiny_bert_pred.to_csv(f"{SAVE_PATH}/part1_tiny_bert_model_test_prediction.csv", index=False)

"""**Your trianing code here...**

If your prediction is saved in pandas dataframe, you can do something like:
```
if __name__ == '__main__':
   part1_tiny_bert_pred.to_csv(f"{SAVE_PATH}/part1_tiny_bert_model_test_prediction.csv", index=False)
```

## Part 1.b

For this section, choose a different pre-trained BERT-style model from the [Hugging Face Model Hub](https://huggingface.co/models) and fine-tune it. There are tons of options - part of the homework is navigating the hub to find different models! I recommend picking a model that is smaller than BERT-Base (as TinyBERT is) just to make things computationally cheaper. Is the final validation accuracy higher or lower with this other model?

**TODO**
* As in part 1.a, save model predictions on the test set to *part1_hf_bert_model_test_prediction.csv*. The csv file should contain "index" columns, corresponding to the unique sample index, and "pred" column, the model prediction on that sample. Your model should achieve >=80% on the test accuracy to receive a full mark.
"""

if __name__ == '__main__':
    ############### YOUR CODE ###############
    # TODO: find a new HF BERT based model from HuggingFace and load it.
    HF_BERT_BASED_MODEL = "microsoft/deberta-v3-small"

    dataset = load_dataset('r-three/ag_news_subset')
    model = AutoModelForSequenceClassification.from_pretrained(HF_BERT_BASED_MODEL, num_labels=4)
    tokenizer = AutoTokenizer.from_pretrained(HF_BERT_BASED_MODEL)
    print(dataset["train"][100])
    #########################################

if __name__ == '__main__':
  tokenized_dataset = dataset.map(tokenize, batched=True)
  metric = evaluate.load('accuracy')

  training_args = TrainingArguments(
      output_dir='part1_tiny_bert_pred',
      eval_strategy='epoch',
      report_to='none'
  )

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_dataset['train'],
      eval_dataset=tokenized_dataset['validation'],
      compute_metrics=compute_metrics,
      data_collator=data_collator
  )

  trainer.train()

  predictions = trainer.predict(tokenized_dataset['test'].remove_columns(['label']))
  part1_hf_bert_pred = pd.DataFrame({
      'index': dataset['test']['index'],
      'pred': np.argmax(predictions.predictions, axis=1)
  })
  part1_hf_bert_pred.to_csv(f"{SAVE_PATH}/part1_hf_bert_model_test_prediction.csv", index=False)

"""**Your training code here...**

Similarly, you can consider something like:

```
if __name__ == '__main__':
   part1_hf_bert_pred.to_csv(f"{SAVE_PATH}/part1_hf_bert_model_test_prediction.csv", index=False)
```

# Part 2 (2.5 points)

Instead of fine-tuning the full model on a target dataset, it's also possible to use the output representations from a BERT-style model as input to a linear classifier and *only* train the classifier (leaving the rest of the pre-trained parameters fixed). You can do this easily using the [`sentence-transformers`](https://www.sbert.net/) library. Using `sentence-tranformers` gives you back a fixed-length representation of a given text sequence. To achieve this, you need to
1. Pick a pre-trained sentence Transformer.
2. Load the AG News dataset and feed the text from each example into the model.
3. Train a linear classifier on the representations.
4. Evaluate performance on the validation set.

For the second step, you can learn more about how to use Hugging Face datasets [here](https://huggingface.co/docs/datasets/index). For the third and fourth step, it's possible to either do this directly in PyTorch, or collect the learned representations and use them as feature vectors to train a linear classifier in any other library (e.g. [scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html)). For this homework, you will implement the second approach.

After you complete the above steps, is the accuracy on the validation set higher or lower using a fixed sentence Transformer?

**TODO**:
* Complete the `encode_data` function: the function embeds each text sample into an output representation using the provided sentence encoder. The function is called to map a text data sample to the model representation, as shown below:
```
dataset.map(lambda x: encode_data(sen_model, x), batched=True)
```
* Train a Logistic Regression classifier: use sklearn.linear_model.LogisticRegression to fit the model on the encoded text data.
* Save your trained model: After training, saved teh fitted logistic regression model as `sentence_encoder_classification.pkl`. Your model should achieve >=85% on the test accuracy to receive a full mark.
"""

def encode_data(model, x):
    """Takes the model and the dataset object
        Returns a dictionary consisting of "encoded_input" and "label" as keys.
        - "encoded_input" contains the tokenized text features produced by the sentence transformer.
        - "label" is the target class label for each example.
        encoded_input is the encoded text input, and label is the target label.
        NOTE: Please assume the dataset object is the original one loaded via
              load_dataset('r-three/') for reproducibility.
              Which means if you want to create additional features to create the encoded_input,
              do so within this function.
    """
    ####################### YOUR CODE ##########################
    # TODO: encoded_input
    concat = []
    for i in range(len(x['title'])):
      concat.append(x['title'][i] + '-' + x['description'][i])
    d = {'encoded_input': model.encode(concat), 'label': x['label']}
    return d
    ############################################################

########### PUT YOUR MODEL HERE ###########
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
###########################################

########### DO NOT CHANGE THIS CODE ###########
if __name__ == "__main__":

    sen_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    # Prepare the dataset
    tokenized_dataset = dataset.map(lambda x: encode_data(sen_model, x), batched=True)
    print(tokenized_dataset['train'][100])
    X_train = np.stack([np.array(x['encoded_input']) for x in tokenized_dataset['train']])
    X_val = np.stack([np.array(x['encoded_input']) for x in tokenized_dataset['validation']])
    y_train = np.stack([np.array(x['label']) for x in tokenized_dataset['train']])
    y_val = np.stack([np.array(x['label']) for x in tokenized_dataset['validation']])

    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)

########### COMPLETE THE FOLLOWING LOGISTIC REGRESSION CODE ###########
if __name__ == "__main__":
    classifier = LogisticRegression(max_iter=1000)
    # Your logistic regression training code here

    classifier.fit(X_train, y_train)

######################## TO SUBMIT ########################
if __name__ == "__main__":
    joblib.dump(classifier, f"{SAVE_PATH}/sentence_encoder_classification.pkl")
    # test if it loads as expected
    # loaded_model = joblib.load(f"{SAVE_PATH}/sentence_encoder_classification.pkl")
