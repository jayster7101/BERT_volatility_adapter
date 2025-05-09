from transformers import BertModel
import torch.nn as nn

class StockPriceActionPrediction(nn.Module):
    def __init__(self, hidden_dim):
        super(StockPriceActionPrediction, self).__init__() # call constructor from nn
        self.bert = BertModel.from_pretrained("bert-base-uncased")

         # Freeze BERT layers since training an adapter only
        for param in self.bert.parameters():
            param.requires_grad = False

        # Down-projection layer (reduce dimensionality)
        self.down_projection = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        
        # Up-projection layer (output a single scalar for regression)
        self.up_projection = nn.Linear(hidden_dim, 3)  # Single scalar output
        
        # ReLU activation (non-linearity)
        self.relu = nn.ReLU()

        # Dropout (helps to not have overdependence on certain nodes)
        self.dropout = nn.Dropout(.2)

    '''
    input_ids is the tokenized rendition of the input values and is an array of tokens
    attention_mask: is an output of the tokenizer, its a result of the bert model expecting a fixed length, so the attention mask acts as the padding 
    '''
    
    def forward(self, input_ids, attention_mask): # Defines the actual inference part 
        x = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0] # extracts the [CLS] token embedding,
        x = self.down_projection(x)
        x = self.relu(x)               # 🔥 Non-linearity between layers
        x = self.dropout(x)
        x = self.up_projection(x)
        return x
