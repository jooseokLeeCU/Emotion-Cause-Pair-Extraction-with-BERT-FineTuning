# Imports libraries
import torch
from transformers import AutoTokenizer
from typing import Dict, List
import random
from tqdm.autonotebook import tqdm
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from transformers import BertTokenizer
from transformers import BertModel, AdamW
from torch import nn

class SentimentDataBert(Dataset):
    def __init__(self, list_utterance, list_emotion):
        # Mapping emotions to index
        self.label_dict = {'joy': 0, 'sadness': 1, 'disgust': 2, 'fear': 3, 'anger': 4, 'neutral': 5, 'surprise': 6}

        self.list_utterance = list_utterance
        self.list_emotion = list_emotion

        # BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Load Dataset
        self.data = self.load_data()

    def load_data(self):
        MAX_LEN = 512  # Maximum input length of BERT
        token_ids = []
        mask_ids = []
        y = []

        for utterance, emotion in zip(self.list_utterance, self.list_emotion):
            encoded_dict = self.tokenizer.encode_plus(
                utterance,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding=True,
                return_attention_mask=True,
                truncation=True,
                return_tensors='pt',
            )

            token_ids.append(encoded_dict['input_ids'])
            mask_ids.append(encoded_dict['attention_mask'])
            y.append(self.label_dict[emotion])

        token_ids = torch.cat(token_ids, dim=0)
        mask_ids = torch.cat(mask_ids, dim=0)
        y = torch.tensor(y)
        return TensorDataset(token_ids, mask_ids, y)

    def get_data_loaders(self, batch_size=32, shuffle=True):
        data_loader = DataLoader(
            self.data,
            shuffle=shuffle,
            batch_size=batch_size
        )
        return data_loader

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

def evaluate_model(model, data_loader, device='cpu'):
    """
    Evaluates the model on the validation or test set.
    
    Args:
        model (torch.nn.Module): The trained model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation/test dataset.
        device (str): 'cuda' or 'cpu' specifying the device to evaluate on.

    Returns:
        predictions (list): List of predicted labels.
        true_labels (list): List of true labels.
    """
    model.eval()  # This disables dropout layers and batch norm updates during inference

    # Lists to store predictions and true labels
    predictions = []
    true_labels = []

    # Disable gradient calculation for inference (saves memory and computation)
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            # Forward pass through the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get predicted class by selecting the class with the highest score (logits)
            _, preds = torch.max(outputs, dim=1)

            # Append predictions and true labels
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels