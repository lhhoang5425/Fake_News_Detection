# Nhan Dien Tin Tuc Gia Bang PyTorch (So Sanh RNN vs LSTM vs GRU)

Du an Hoc Sau (Deep Learning) toan dien ve bai toan phan loai van ban: Xac thuc Tin That (Real News) hay Tin Gia (Fake News) su dung thu vien PyTorch thuan. 

Du an xay dung mot quy trinh hoan chinh tu du lieu tho, tien xu ly, thiet ke kien truc mang no-ron hoi quy, huan luyen danh gia mo hinh, cho den viec dong goi thanh cac cong cu dong lenh (CLI) de dang tai hien tren bat ky may tinh nao.

---

## 1. Tong Quan Bai Toan

Trong thoi dai so hoa va mang xa hoi, tin gia (Fake News / Misinformation) xuat hien tran lan voi noi dung giat gan, bop meo su that nham muc dich cau view, thao tung du luan hoac lua dao.

Du an nay giai quyet bai toan phan loai nhi phan (Binary Classification):
- Nhan 0 (Real News - Tin That): Bai bao chinh thong, da qua kiem duyet thuc te.
- Nhan 1 (Fake News - Tin Gia): Bai viet sai lech, khong co can cu xac thuc.

Muc tieu cot loi: So sanh thuc nghiem xem giua Simple RNN, LSTM va GRU, kien truc nao co kha nang hoc va ghi nho ngu canh van ban dai tot nhat de ngan chan tin gia hieu qua.

---

## 2. Cau Truc Thu Muc Du An

```text
Fake-News-Detection-PyTorch/
├── data/
│   └── Fake_News_Detection_Dataset.csv   # Dataset chua noi dung bai bao va nhan (0/1)
├── notebooks/                            # Thu nghiem nghien cuu chi tiet tung mo hinh
│   ├── 1_Fake_News_RNN.ipynb             # Thi nghiem mo hinh Simple RNN
│   ├── 2_Fake_News_LSTM.ipynb            # Thi nghiem mo hinh LSTM
│   └── 3_Fake_News_GRU.ipynb             # Thi nghiem mo hinh GRU
├── src/                                  # Ma nguon dang module chuan ky su
│   ├── preprocess.py                     # Lam sach text, tao Vocab, chuyen Sequence & Dataset
│   ├── models.py                         # Dinh nghia 3 kien truc: RNNClassifier, LSTMClassifier, GRUClassifier
│   ├── train.py                          # Script CLI huan luyen va danh gia tren tap Test
│   └── predict.py                        # Script CLI du doan cau bai bao moi
├── saved_models/                         # Luu tru trong so mo hinh va tu dien sau khi train
│   ├── vocab.pkl                         # File tu dien Vocab da huan luyen
│   └── best_lstm_model.pth               # Checkpoint trong so mo hinh tot nhat (LSTM)
├── requirements.txt                      # Danh sach thu vien can thiet
├── .gitignore                            # Bo qua file rac, bo nho cache
└── README.md                             # Tai lieu huong dan du an
```

---

## 3. Phan Tich Ky Thuat Tung Buoc (Pipeline Deep Learning)

### Buoc 1: Lam Sach Van Ban (Text Cleaning)
- Van de: Van ban tho tren Internet chua nhieu dau cau, ky tu dac biet, chu hoa chu thuong lan lon khien mo hinh bi nhieu.
- Giai phap:
  - Chuyen toan bo ky tu ve chu thuong (lower()).
  - Dung Bieu thuc chinh quy (Regex) `[^a-z0-9\s]` de loc bo toan bo ky tu la, chi giu lai chu cai alphabet va so.

### Buoc 2: Xay Dung Bo Tu Dien (Vocabulary) & Ma Hoa So Hoc
- Van de: Mang no-ron chi tinh toan duoc voi ma tran so hoc (Tensor), khong hieu duoc chuoi ky tu.
- Giai phap:
  - Dem tan suat xuat hien cua tu bang collections.Counter.
  - Chon ra max_vocab_size = 10,000 tu pho bien nhat va gan cho moi tu mot chi so nguyen (index).
  - Them 2 Token dac biet:
    - `<PAD>` (index 0): Dung de chen them so 0 vao cac cau ngan cho du do dai chuan.
    - `<UNK>` (index 1): Dai dien cho cac tu hiem gap hoac tu moi khong co trong tu dien.
  - Cat ngan (Truncation) hoac bu do dai (Padding) dua moi cau ve do dai co dinh max_len = 150 tu.

### Buoc 3: Tang Nhung Tu (Embedding Layer)
- Thay vi su dung One-Hot Encoding gay ton bo nho va roi rac, tang `nn.Embedding(vocab_size, embed_dim=128)` se anh xa moi tu thanh mot vector lien tuc 128 chieu.
- Nho do, cac tu co ngu nghia tuong dong trong khong gian vector se nam gan nhau.

### Buoc 4: Thiet Ke & So Sanh 3 Kien Truc Mang No-ron
1. Simple RNN (nn.RNN):
   - Nguyen ly: Truyen hidden state tu tu phia truoc sang tu phia sau theo thu tu thoi gian.
   - Han che: Gap hien tuong Vanishing Gradient (Triet tieu dao ham). Khi cau dai den tu thu 50-100, mo hinh gan nhu quen sach thong tin o nhung tu dau cau.
2. LSTM (nn.LSTM - Khuyen dung):
   - Nguyen ly: Bo sung Cell State (bo nho dai han) chay xuyen suot chuoi va kiem soat bang 3 cong logic (Gates):
     - Forget Gate (Cong quen): Quyet dinh thong tin nao trong qua khu khong con quan trong va can xoa bo.
     - Input Gate (Cong nap): Quyet dinh thong tin moi nao can ghi nho vao Cell State.
     - Output Gate (Cong xuat): Quyet dinh thong tin nao duoc dua ra hidden state tiep theo.
   - Ket qua: Duy tri duoc ngu canh cua ca bai bao dai.
3. GRU (nn.GRU):
   - Nguyen ly: Bien the rut gon cua LSTM, chi su dung 2 cong (Reset Gate va Update Gate).
   - Ket qua: Huan luyen nhanh hon LSTM, ton it tai nguyen hon ma hieu nang van rat sat sao.

### Buoc 5: Ham Mat Mat & Toi Uu Hoa
- Ham mat mat (Loss Function): Su dung `nn.BCELoss` (Binary Cross Entropy) ket hop ham kich hoat Sigmoid o ngo ra de dua xac suat ve khoang [0.0, 1.0].
- Nguong quyet dinh (Threshold): Xac suat >= 0.5 -> Tin Gia (Fake), < 0.5 -> Tin That (Real).
- Thuat toan toi uu: Adam voi learning rate 0.001.

---

## 4. Bang Danh Gia Ket Qua Thuc Nghiem Chi Tiet

Ket qua do luong tren cung tap kiem thu doc lap (Test Set) khong tham gia vao qua trinh huan luyen:

| Kien Truc Mo Hinh | Do Chinh Xac (Accuracy) | F1-Score | Precision (Tin Gia) | Recall (Tin Gia) | Thoi Gian Train / Epoch | Nhan Xet Danh Gia |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Simple RNN** | `62.00%` | `0.60` | `0.73` | `0.39` | Nhanh nhat (~1s) | **Kem nhat:** Bi triet tieu dao ham, bo lot hon 60% tin gia (Recall chi dat 39%). |
| **GRU** | `82.00%` | `0.83` | `0.81` | `0.85` | Trung binh (~2s) | **Can bang tot:** Hoi tu nhanh, phat hien chuan xac ca tin that lan tin gia. |
| **LSTM (Tot nhat)** | **`85.00%`** | **`0.86`** | **`0.80`** | **`0.92`** | ~2.5s | **Vuot troi:** Kha nang ghi nho ngu canh bai bao dai rat tot, phat hien duoc **92%** so luong tin gia. |

---

## 5. Danh Gia & Ket Luan Chuyen Mon

1. Tai sao Recall cua Tin Gia la chi so quan trong nhat?
   - Trong bai toan phat hien tin tuc gia hoac lua dao, viec doan nham tin gia thanh tin that (False Negative) nguy hiem hon nhieu so voi viec nghi ngo nham mot tin that.
   - Mo hinh LSTM dat chi so Recall 92% cho lop Tin Gia, chung minh co che cong cua LSTM la cuc ky hieu qua de khong bo sot cac thong tin sai lech.
2. So sanh danh doi (Trade-off):
   - Neu can mo hinh nhe, trien khai tren thiet bi cau hinh yeu hoac he thong thoi gian thuc voi luong truy cap lon: GRU la lua chon toi uu ve chi phi tinh toan.
   - Neu uu tien do chinh xac va do nhay toi da de bao ve nguoi dung: LSTM la mo hinh chien thang.

---

## 6. Huong Dan Cai Dat & Chay Kiem Thu (Cho Moi Nguoi Dung)

### 1. Cai dat thu vien
Mo Terminal / Command Prompt tai thu muc du an va chay:

```bash
pip install -r requirements.txt
```

### 2. Kiem thu doc lap tung thanh phan (Unit Test)
Moi module deu duoc thiet ke doc lap, ban co the chay de xem co che hoat dong:

- Kiem tra bo tien xu ly va chuyen chuoi van ban:
  ```bash
  python src/preprocess.py
  ```

- Kiem tra luong tinh toan ma tran cua 3 mo hinh:
  ```bash
  python src/models.py
  ```

### 3. Huan luyen mo hinh (Training qua CLI)
Ban co the tu do huan luyen lai mo hinh mong muon chi bang 1 cau lenh:

```bash
# Huan luyen mo hinh LSTM (Khuyen dung)
python src/train.py --model lstm --epochs 10 --batch_size 64

# Huan luyen mo hinh GRU
python src/train.py --model gru --epochs 10 --batch_size 64

# Huan luyen mo hinh Simple RNN (Baseline)
python src/train.py --model rnn --epochs 10 --batch_size 64
```
Mo hinh co chi so Validation Accuracy cao nhat se tu dong duoc luu vao thu muc `saved_models/`.

### 4. Du doan bai bao bat ky (Inference)
Kiem tra kha nang phan doan cua mo hinh bang cach truyen cau bat ky qua tham so `--text`:

```bash
python src/predict.py --model lstm --text "Scientists discover ancient water reservoir deep beneath the surface of Mars."
```

Ket qua hien thi tren Terminal:
```text
==================================================
KET QUA PHAN LOAI BAI BAO:
Noi dung: "Scientists discover ancient water reservoir deep beneath the surface of Mars."
Phan loai: TIN THAT (Real News)
Do tin cay (Confidence): 89.25% (Xac suat raw: 0.1075)
==================================================
```

Hoac chay che do hoi dap tuong tac truc tiep:
```bash
python src/predict.py
```

---

## 7. Cac Cong Nghe & Ky Thuat Su Dung

- Ngon ngu: Python 3.9+
- Deep Learning Framework: PyTorch (torch, torch.nn, torch.optim, Dataset, DataLoader)
- Xu ly Du lieu: Pandas, NumPy, Regex (re), Collections (Counter)
- Khoa hoc Du lieu & Danh gia: Scikit-learn (train_test_split, classification_report, f1_score, accuracy_score)
- Truc quan hoa: Matplotlib, Seaborn
- Luu tru & Dong goi: Pickle, PyTorch Checkpoints (.pth), Argparse CLI
