import torch
from torch import nn

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class myModle(nn.Module):
    def __init__(self, words_num, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        Embedding = nn.Embedding(num_embeddings=words_num, embedding_dim=50)
        LSTM = nn.LSTM(input_size=50, hidden_size=self.hidden_size, num_layers=4, batch_first=True)
        Linear = nn.Linear(self.hidden_size, words_num)

        self.add_module("Embedding", Embedding)
        self.add_module("LSTM", LSTM)
        self.add_module("Linear", Linear)

    def forward(self, x: torch.Tensor):
        x = x.to(torch.int64)
        x = self.Embedding(x)
        h0 = torch.zeros(4, x.shape[0], self.hidden_size, device=device)
        c0 = torch.zeros(4, x.shape[0], self.hidden_size, device=device)
        out, (_, _) = self.LSTM(x, (h0, c0))
        out = self.Linear(out)
        return out




