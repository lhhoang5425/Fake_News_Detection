import os
import argparse
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text, build_vocab, text_to_sequence, FakeNewsDataset
from models import RNNClassifier, LSTMClassifier, GRUClassifier

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dang su dung thiet bi: {device}")
    
    # 1. Doc du lieu
    # Xu ly duong dan an toan du dung o thu muc goc hay trong src/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(base_dir, args.data_path)
    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(base_dir, args.save_dir)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Khong tim thay file dataset tai: {data_path}")
        
    print(f"Dang doc dataset tu: {data_path}")
    df = pd.read_csv(data_path)
    df['clean_text'] = df['text'].apply(clean_text)
    
    # 2. Chia tap Train, Val, Test (70% - 15% - 15%)
    X_train_val, X_test_raw, y_train_val, y_test_raw = train_test_split(
        df['clean_text'].values, df['label'].values, test_size=0.15, random_state=42, stratify=df['label']
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
    )
    
    # 3. Xay dung Vocab tren tap Train
    vocab = build_vocab(X_train_raw, max_vocab_size=args.max_vocab)
    print(f"Kich thuoc tu dien (Vocab size): {len(vocab)}")
    
    # Luu tu dien de dung cho buoc inference sau nay
    os.makedirs(save_dir, exist_ok=True)
    vocab_file = os.path.join(save_dir, 'vocab.pkl')
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Da luu tu dien tai: {vocab_file}")
    
    # 4. Chuyen doi van ban thanh sequence tensors
    X_train = [text_to_sequence(t, vocab, args.max_len) for t in X_train_raw]
    X_val = [text_to_sequence(t, vocab, args.max_len) for t in X_val_raw]
    X_test = [text_to_sequence(t, vocab, args.max_len) for t in X_test_raw]
    
    train_loader = DataLoader(FakeNewsDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FakeNewsDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FakeNewsDataset(X_test, y_test_raw), batch_size=args.batch_size, shuffle=False)
    
    # 5. Khoi tao mo hinh theo lua chon
    model_type = args.model.lower()
    if model_type == 'lstm':
        model = LSTMClassifier(len(vocab), args.embed_dim, args.hidden_dim).to(device)
    elif model_type == 'gru':
        model = GRUClassifier(len(vocab), args.embed_dim, args.hidden_dim).to(device)
    else:
        model = RNNClassifier(len(vocab), args.embed_dim, args.hidden_dim).to(device)
        
    print(f"Khoi tao kien truc: {model_type.upper()}")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 6. Vong lap huan luyen
    best_val_acc = 0.0
    model_save_path = os.path.join(save_dir, f'best_{model_type}_model.pth')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y_b)
            predicted_labels = (preds >= 0.5).float()
            correct += (predicted_labels == y_b).sum().item()
            total += len(y_b)
            
        train_acc = correct / total
        
        # Danh gia tren tap Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b)
                val_correct += ((preds >= 0.5).float() == y_b).sum().item()
                val_total += len(y_b)
        val_acc = val_correct / val_total
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            
    print(f"Huan luyen hoan tat! Trong so mo hinh tot nhat luu tai: {model_save_path}")
    
    # 7. Danh gia cuoi cung tren tap Test
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            preds = model(X_b)
            test_preds.extend((preds >= 0.5).float().cpu().numpy())
            test_targets.extend(y_b.numpy())
            
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds)
    print("\n" + "="*50)
    print(f"KET QUA TREN TAP KIEM THU (TEST SET) - {model_type.upper()}:")
    print(f"Accuracy: {test_acc*100:.2f}% | F1-Score: {test_f1:.4f}")
    print("="*50)
    print(classification_report(test_targets, test_preds, target_names=['Tin That (Real)', 'Tin Gia (Fake)']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fake News Classifier")
    parser.add_argument('--data_path', type=str, default='data/Fake_News_Detection_Dataset.csv')
    parser.add_argument('--model', type=str, default='lstm', choices=['rnn', 'lstm', 'gru'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=150)
    parser.add_argument('--max_vocab', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='saved_models')
    
    args = parser.parse_args()
    train_model(args)
