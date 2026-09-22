import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, output_dim=1):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, h_n = self.rnn(embedded)
        # Lấy hidden state cuối cùng
        last_hidden = h_n.squeeze(0)
        out = self.fc(last_hidden)
        return self.sigmoid(out)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, output_dim=1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, (h_n, c_n) = self.lstm(embedded)
        last_hidden = h_n[-1]
        out = self.fc(last_hidden)
        return self.sigmoid(out)

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, output_dim=1):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, h_n = self.gru(embedded)
        last_hidden = h_n[-1]
        out = self.fc(last_hidden)
        return self.sigmoid(out)

if __name__ == '__main__':
    print("--- [TEST ĐỘC LẬP] MODULE MODELS.PY ---")
    dummy_vocab_size = 1000
    dummy_input = torch.randint(0, dummy_vocab_size, (4, 20)) # Batch 4 câu, mỗi câu 20 từ
    print(f"Kích thước tensor đầu vào giả lập: {dummy_input.shape}")
    
    # Test RNN
    rnn = RNNClassifier(dummy_vocab_size)
    print(f"1. RNN Output shape: {rnn(dummy_input).shape}")
    
    # Test LSTM
    lstm = LSTMClassifier(dummy_vocab_size)
    print(f"2. LSTM Output shape: {lstm(dummy_input).shape}")
    
    # Test GRU
    gru = GRUClassifier(dummy_vocab_size)
    print(f"3. GRU Output shape: {gru(dummy_input).shape}")
    print("✅ Cả 3 kiến trúc mô hình đều biên dịch và tính toán Forward thành công!")
