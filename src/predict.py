import os
import sys
import argparse
import pickle
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text, text_to_sequence
from models import RNNClassifier, LSTMClassifier, GRUClassifier

def predict_text(text, model_type='lstm', model_dir='saved_models', embed_dim=128, hidden_dim=64, max_len=150):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = model_dir if os.path.isabs(model_dir) else os.path.join(base_dir, model_dir)
    
    # 1. Tai Vocab
    vocab_path = os.path.join(target_dir, 'vocab.pkl')
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Khong tim thay file tu dien tai: {vocab_path}. Vui long chay train.py truoc!")
        
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    # 2. Khoi tao va tai trong so mo hinh
    model_path = os.path.join(target_dir, f'best_{model_type.lower()}_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Khong tim thay file mo hinh tai: {model_path}. Vui long chay: python src/train.py --model {model_type.lower()} truoc!")
        
    if model_type.lower() == 'lstm':
        model = LSTMClassifier(len(vocab), embed_dim, hidden_dim)
    elif model_type.lower() == 'gru':
        model = GRUClassifier(len(vocab), embed_dim, hidden_dim)
    else:
        model = RNNClassifier(len(vocab), embed_dim, hidden_dim)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Tien xu ly van ban dau vao
    cleaned = clean_text(text)
    seq = text_to_sequence(cleaned, vocab, max_len=max_len)
    input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
    
    # 4. Du doan
    with torch.no_grad():
        prob = model(input_tensor).item()
        
    label = "TIN GIA (Fake News)" if prob >= 0.5 else "TIN THAT (Real News)"
    confidence = prob if prob >= 0.5 else (1.0 - prob)
    
    return label, confidence, prob

def interactive_mode():
    print("\n" + "="*60)
    print("CHUONG TRINH PHAN LOAI TIN TUC TUONG TAC")
    print("="*60)
    print("Chon mo hinh ban muon su dung:")
    print("  1. LSTM (Khuyen dung - Do chinh xac cao nhat)")
    print("  2. GRU  (Hoi tu nhanh, can bang tot)")
    print("  3. RNN  (Mo hinh hoi quy co ban)")
    print("  0. Thoat chuong trinh")
    print("="*60)
    
    choice_map = {'1': 'lstm', '2': 'gru', '3': 'rnn'}
    current_model = 'lstm'
    
    while True:
        choice = input("\nChon mo hinh (1/2/3) [Mac dinh la 1]: ").strip()
        if choice == '0' or choice.lower() in ['exit', 'quit']:
            print("Da thoat chuong trinh.")
            return
        if choice in choice_map:
            current_model = choice_map[choice]
            break
        elif choice == '':
            current_model = 'lstm'
            break
        else:
            print("Lua chon khong hop le, vui long nhap 1, 2 hoac 3.")

    print(f"\nDa chon mo hinh: [{current_model.upper()}]")
    print("- Nhap/Dan mot cau bai bao vao de kiem tra.")
    print("- Go 'switch' de doi sang mo hinh khac.")
    print("- Go 'exit' hoac 'quit' de thoat.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input(f"\n[{current_model.upper()}] Nhap cau bai bao: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nDa thoat chuong trinh. Hen gap lai!")
                break
            if user_input.lower() == 'switch':
                interactive_mode()
                break
                
            label, conf, prob = predict_text(user_input, model_type=current_model)
            print("-" * 50)
            print(f"KET QUA: {label}")
            print(f"Do tin cay: {conf*100:.2f}% (Xac suat raw: {prob:.4f})")
            print("-" * 50)
        except KeyboardInterrupt:
            print("\nDa dung chuong trinh.")
            break
        except Exception as e:
            print(f"Co loi xay ra: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Fake News on New Text")
    parser.add_argument('--text', type=str, default=None, help="Doan van ban bai bao can phan loai")
    parser.add_argument('--model', type=str, default='lstm', choices=['rnn', 'lstm', 'gru'])
    parser.add_argument('--model_dir', type=str, default='saved_models')
    
    args = parser.parse_args()
    
    if args.text:
        label, conf, prob = predict_text(args.text, args.model, args.model_dir)
        print("\n" + "="*50)
        print(f"KET QUA PHAN LOAI BAI BAO ({args.model.upper()}):")
        print(f"Noi dung: \"{args.text[:100]}...\"" if len(args.text) > 100 else f"Noi dung: \"{args.text}\"")
        print(f"Phan loai: {label}")
        print(f"Do tin cay (Confidence): {conf*100:.2f}% (Xac suat raw: {prob:.4f})")
        print("="*50 + "\n")
    else:
        interactive_mode()
