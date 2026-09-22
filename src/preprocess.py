import re
import torch
from torch.utils.data import Dataset
from collections import Counter

def clean_text(text):
    """
    Làm sạch văn bản:
    - Chuyển toàn bộ về chữ thường
    - Loại bỏ ký tự đặc biệt, dấu câu, chỉ giữ lại chữ cái và số
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def build_vocab(texts, max_vocab_size=10000):
    """
    Xây dựng từ điển (Vocabulary) từ danh sách văn bản:
    - Token <PAD> mang index 0 (dùng để bù độ dài câu)
    - Token <UNK> mang index 1 (từ không có trong từ điển)
    - Các từ phổ biến nhất sẽ được gán index từ 2 trở đi
    """
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
        
    most_common = word_counts.most_common(max_vocab_size - 2)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
        
    return vocab

def text_to_sequence(text, vocab, max_len=150):
    """
    Chuyển một câu văn bản thành chuỗi số nguyên có độ dài cố định max_len.
    - Cắt bớt nếu câu quá dài (Truncation)
    - Bù số 0 nếu câu quá ngắn (Padding)
    """
    words = text.split()
    seq = [vocab.get(word, vocab.get('<UNK>', 1)) for word in words]
    
    if len(seq) < max_len:
        seq = seq + [vocab['<PAD>']] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

class FakeNewsDataset(Dataset):
    """
    PyTorch Dataset tùy chỉnh để đóng gói X (sequences) và y (nhãn 0/1)
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == '__main__':
    print("--- [TEST ĐỘC LẬP] MODULE PREPROCESS.PY ---")
    sample_text = "Breaking News: Officials Announce New AI Regulations in 2026!"
    print(f"1. Văn bản gốc: {sample_text}")
    cleaned = clean_text(sample_text)
    print(f"2. Sau khi làm sạch: {cleaned}")
    vocab = build_vocab([cleaned], max_vocab_size=100)
    print(f"3. Từ điển mẫu (Vocab): {vocab}")
    seq = text_to_sequence(cleaned, vocab, max_len=12)
    print(f"4. Chuyển thành chuỗi số (Sequence, max_len=12): {seq}")
    print("✅ Module preprocess.py hoạt động hoàn hảo!")
