import os
import pandas as pd
import numpy as np
import json
import re
from nltk.tokenize import sent_tokenize 
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split
import datetime 
import warnings
warnings.filterwarnings('ignore')
import nltk

# Load the dataset
ner_df = pd.read_csv('new_ds.csv')
ner_df['Entities'] = ner_df['Entities'].apply(lambda x: eval(x) if type(x)==str else x)
train, val = train_test_split(ner_df, test_size=0.1, random_state=42)

from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
import torch
import accelerate

# Load BERT model and tokenizer

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')



def clean_text(txt):
    '''
    This is text cleaning function
    '''
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())

def data_joining(data_dict_id):
    '''
    This function is to join all the text data from different 
    sections in the json to a single text file. 
    '''
    data_length = len(data_dict_id)

    #     temp = [clean_text(data_dict_id[i]['text']) for i in range(data_length)]
    temp = [data_dict_id[i]['text'] for i in range(data_length)]
    temp = '. '.join(temp)
    
    return temp

def make_shorter_sentence(sentence):
    
    sent_tokenized = sent_tokenize(sentence)
    
    max_length = 128
    overlap = 20
    
    final_sentences = []
    
    for tokenized_sent in sent_tokenized:
        sent_tokenized_clean = clean_text(tokenized_sent)
        sent_tokenized_clean = sent_tokenized_clean.replace('.','').rstrip() 
        
        tok_sent = sent_tokenized_clean.split(" ")
        
        if len(tok_sent)<max_length:
            final_sentences.append(sent_tokenized_clean)
        else :
#             print("Making shorter sentences")
            start = 0
            end = len(tok_sent)
            
            for i in range(start, end, max_length-overlap):
                temp = tok_sent[i: (i + max_length)]
                final_sentences.append(" ".join(i for i in temp))

    return final_sentences
def form_labels(sentence, labels_list):
    '''
    This function labels the training data 
    '''
    matched_kwords = []
    matched_token = []
    un_matched_kwords = []
    label = []
    
    # Since there are many sentences which are more than 512 words,
    # Let's make the max length to be 128 words per sentence.
    tokens = make_shorter_sentence(sentence)
    
    for tok in tokens:    
        tok_split = tokenizer.tokenize(tok)
          
        z = np.array(['O'] * len(tok_split)) # Create final label == len(tokens) of each sentence
        matched_keywords = 0 # Initially no kword matched    

        for kword in labels_list:
            kword = kword.lower()
            
            
            if kword in tok: #This is to first check if the keyword is in the text and then go ahead
                kword_split = tokenizer.tokenize(kword)
                
                for i in range(len(tok_split)):
                    if tok_split[i: (i + len(kword_split))] == kword_split:
                        
                        matched_keywords += 1

                        if (len(kword_split) == 1):
                            z[i] = 'B'
                        else:
                            z[i] = 'B'
                            z[(i+1) : (i+ len(kword_split))]= 'B'

                        if matched_keywords >1:
                            label[-1] = (z.tolist())
                            matched_token[-1] = tok
                            matched_kwords[-1].append(kword)
                        else:
                            label.append(z.tolist())
                            matched_token.append(tok)
                            matched_kwords.append([kword])
                    else:
                        un_matched_kwords.append(kword)
                
    return matched_token, matched_kwords, label, un_matched_kwords
def labelling(dataset):
    '''
    This function is to iterate each of the training data and get it labelled 
    from the form_labels() function.
    '''
    
    
    sentences_ = []
    key_ = []
    labels_ = []
    un_mat = []
    un_matched_reviews = 0
    text = dataset['Text'].to_list()
    labels = dataset['Entities'].to_list()
    for i in range(len(text)):

        sentence = text[i]
        label = labels[i]

        s, k, l, un_matched = form_labels(sentence=sentence, labels_list = label)

        if len(s) == 0:
            un_matched_reviews += 1
            un_mat.append(un_matched)
        else: 
            sentences_.append(s)
            key_.append(k)
            labels_.append(l)
            

    print("Total unmatched keywords:", un_matched_reviews)
    sentences = [item for sublist in sentences_ for item in sublist]
    final_labels = [item for sublist in labels_ for item in sublist]
    keywords = [item for sublist in key_ for item in sublist]
    
    
    return sentences, final_labels, keywords


#train['Entities'] = train['Entities'].apply(lambda x: eval(x) if type(x)==str else x)

train_sentences, train_label, train_keywords = labelling(dataset = train)
label_map = {'B': 1, 'O': 0}
labels_indices = [[label_map[label] for label in sequence] for sequence in train_label]
print("")
print(f" train sentences: {len(train_sentences)}, train label: {len(train_label)}, train keywords: {len(train_keywords)}")
tokenized_input = tokenizer(train_sentences, truncation=True, padding='longest', return_tensors='pt')

max_len = len(tokenized_input['input_ids'][0])
for i in range(len(labels_indices)):
    if len(labels_indices[i]) < max_len:
        labels_indices[i] = labels_indices[i] + ([0]*(max_len-len(labels_indices[i])))

val_sentences, val_label, val_keywords = labelling(dataset = val)
labels_indices_val = [[label_map[label] for label in sequence] for sequence in val_label]
print("")
print(f" val sentences: {len(val_sentences)}, val label: {len(val_label)}, val keywords: {len(val_keywords)}")

tokenized_input_val = tokenizer(val_sentences, truncation=True, padding='longest', return_tensors='pt')

max_len = len(tokenized_input_val['input_ids'][0])

for i in range(len(labels_indices_val)):
    if len(labels_indices_val[i]) < max_len:
        labels_indices_val[i] = labels_indices_val[i] + ([0]*(max_len-len(labels_indices_val[i])))
unique_df = pd.DataFrame({
                          'train_sentences': train_sentences, 
                          'kword': train_keywords, 
                          'label':labels_indices})
#unique_df.label = unique_df.label.astype('str')
#unique_df.kword = unique_df.kword.astype('str')
#unique_df['sent_len'] = unique_df.train_sentences.apply(lambda x : len(x.split(" ")))
#unique_df = unique_df.drop_duplicates()
#print(unique_df.shape)
print(unique_df)
#tokenized_input['input_ids'][0] + torch.Tensor([0]*3)
#len(tokenizer(train_sentences, truncation=True, padding='longest', return_tensors='pt')['input_ids'][0])
from torch.utils.data import DataLoader, TensorDataset

# Assuming tokenized_input is a dictionary with 'input_ids', 'attention_mask', and 'labels'
dataset = TensorDataset(tokenized_input['input_ids'], tokenized_input['attention_mask'], torch.tensor(labels_indices))

# DataLoader
train_dataloader = DataLoader(dataset, shuffle=True)
epochs = 10
total_steps = len(train_dataloader) * epochs
from transformers import  AdamW, get_linear_schedule_with_warmup

model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(ner_df['Entities']))
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=0).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_epoch_accuracy = 0
    for batch in train_dataloader:
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}
        
        (loss, logits) = model(batch[0].to(device), 
                                token_type_ids=None, 
                                attention_mask=batch[1].to(device),
                                labels=batch[2].to(device))
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        total_epoch_accuracy += flat_accuracy(logits, batch[2].to(device).to('cpu').numpy())
        loss.backward()
        optimizer.step()
        scheduler.step()
    avg_epoch_accuracy = total_epoch_accuracy / len(train_dataloader)
    average_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}, Accuracy: {avg_epoch_accuracy}')

dataset = TensorDataset(tokenized_input_val['input_ids'], tokenized_input_val['attention_mask'], torch.tensor(labels_indices_val))
# DataLoader
val_dataloader = DataLoader(dataset, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    total_eval_accuracy = 0
    for batch in val_dataloader:
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}
        
        (loss, logits) = model(batch[0].to(device), 
                                token_type_ids=None, 
                                attention_mask=batch[1].to(device),
                                labels=batch[2].to(device))
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

        all_predictions.extend(predictions.cpu().numpy().flatten())
        all_targets.extend(inputs['labels'].cpu().numpy().flatten())
        total_eval_accuracy += flat_accuracy(logits, batch[2].to(device).to('cpu').numpy())
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))


model.save_pretrained("model")
tokenizer.save_pretrained("model/tokenizer")
