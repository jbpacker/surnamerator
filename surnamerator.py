import torch
import torch.nn as nn

## https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
class MLP(nn.Module):
    def __init__(self, num_char, hidden_nodes, embeddings, window, num_layers):   
        super(MLP, self).__init__()
        
        self.window = window
        self.hidden_nodes = hidden_nodes
        self.embeddings = embeddings        
        
        self.layers = nn.Sequential(
            nn.Embedding(num_char, embeddings),         # [batch, window] --> [batch, window, embeddings]
            nn.Flatten(),                               # [batch, window, embeddings] --> [batch, window*embeddings]
            nn.Linear(embeddings*window, hidden_nodes), # [batch, window*embeddings] --> [batch, hidden_nodes]
            
        )
        for i in range(num_layers):
            self.layers = self.layers.extend(nn.Sequential(
                nn.Linear(hidden_nodes, hidden_nodes, bias=False),
                nn.BatchNorm1d(hidden_nodes),
                nn.Tanh()))
            
        self.layers = self.layers.extend(nn.Sequential(
            nn.Linear(hidden_nodes, num_char)
        ))
        
        with torch.no_grad():
            self.layers[-1].weight *= 0.1           
        
    def forward(self, x):
        return self.layers(x)

# Follows this paper: https://arxiv.org/abs/1609.03499
class WaveNet(nn.Module):
    def __init__(self, num_char, hidden_nodes, embeddings, window, num_layers):   
        super(WaveNet, self).__init__()
        
        self.window = window
        self.hidden_nodes = hidden_nodes
        self.embeddings = embeddings        
        
        self.layers = nn.Sequential(
            nn.Embedding(num_char, embeddings)
        )
        
        for i in range(num_layers):
            if i == 0:
                nodes = window
            else:
                nodes = hidden_nodes
                
            self.layers = self.layers.extend(nn.Sequential(
                nn.Conv1d(nodes, hidden_nodes, kernel_size=2, stride=1, bias=False),
                nn.BatchNorm1d(hidden_nodes),
                nn.Tanh()))
            
        self.layers = self.layers.extend(nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_nodes*(embeddings-num_layers), num_char)
        ))
        
    def forward(self, x):
        return self.layers(x)