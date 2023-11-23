import torch
from transformers import BertTokenizer, BertForTokenClassification

model_path = "model"
token_path = "model/tokenizer"
tokenizer = BertTokenizer.from_pretrained(token_path)
model = BertForTokenClassification.from_pretrained(model_path)


device = torch.device('cpu')
model.to(device)

text = "Examples of fault-block mountains include the Sierra Nevada in California and Nevada, the Tetons in Wyoming, and the Harz Mountains in Germany."
tokens = tokenizer.encode_plus(text, return_tensors='pt')
with torch.no_grad():
    
    outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=2)

print(predictions)
predicted_entities = [tokenizer.decode(token) for token in predictions[0].tolist()]
print(predicted_entities)
mountain_entities = []

for token, label in zip(tokenizer.tokenize(text), predicted_entities):
    if label != '[ P A D ]':
        mountain_entities.append(token)

mountain_entities = ' '.join(mountain_entities)

print(mountain_entities)