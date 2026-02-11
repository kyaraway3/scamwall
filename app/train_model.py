import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import gc
from tqdm import tqdm
import os  # ★追加: ファイルサイズ確認用

# --- 設定 ---
BATCH_SIZE = 8       
EPOCHS = 3            
LEARNING_RATE = 2e-5  
MAX_LEN = 128         
MODEL_NAME = 'cl-tohoku/bert-base-japanese-v3'

# --- デバイスの自動選択 ---
def get_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPUが見つかりました: {device_name}")
        return torch.device("cuda")
    else:
        print("⚠️ GPUが見つかりません。CPUで学習します（時間がかかります）。")
        return torch.device("cpu")

DEVICE = get_device()

# --- 1. データセット定義 ---
class AppDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.risky_perms = [
            "SYSTEM_ALERT_WINDOW", "RECEIVE_BOOT_COMPLETED", 
            "BIND_ACCESSIBILITY_SERVICE", "READ_CONTACTS"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = str(row['description']) if pd.notna(row['description']) else ""
        perms_str = str(row['permissions']) if pd.notna(row['permissions']) else ""
        perm_vec = [1.0 if rp in perms_str else 0.0 for rp in self.risky_perms]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'metadata': torch.tensor(perm_vec, dtype=torch.float),
            'labels': torch.tensor(row['is_fraud'], dtype=torch.float)
        }

# --- 2. モデル定義 (ハイブリッド型) ---
class FraudDetector(nn.Module):
    def __init__(self, n_meta_features):
        super(FraudDetector, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.meta_layer = nn.Linear(n_meta_features, 32)
        self.out = nn.Linear(768 + 32, 1)

    def forward(self, input_ids, attention_mask, metadata):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        bert_out = self.drop(outputs.pooler_output)
        meta_out = torch.relu(self.meta_layer(metadata))
        combined = torch.cat((bert_out, meta_out), dim=1)
        return self.out(combined)

# --- 3. 学習ループ関数 ---
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(DEVICE)
        attention_mask = d["attention_mask"].to(DEVICE)
        metadata = d["metadata"].to(DEVICE)
        labels = d["labels"].to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                metadata=metadata
            )
            loss = loss_fn(outputs, labels.unsqueeze(1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler:
            scheduler.step()

        preds = torch.sigmoid(outputs).round()
        correct_predictions += torch.sum(preds.flatten() == labels)
        losses.append(loss.item())
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return correct_predictions.double() / n_examples, np.mean(losses)

# --- メイン実行部 ---
def main():
    try:
        # データ読み込み
        print("📂 データを読み込んでいます...")
        # パスは環境に合わせて適宜修正してください
        if os.path.exists(r'C:\learn\scamwall\app_dataset_labeled.csv'):
            csv_path = r'C:\learn\scamwall\app_dataset_labeled.csv'
        else:
            csv_path = 'app_dataset_labeled.csv' # カレントディレクトリ用
            
        df = pd.read_csv(csv_path)
        
        if len(df) < 10:
            print("⚠️ データが少なすぎます。")
            return

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
        
        train_dataset = AppDataset(df_train, tokenizer, MAX_LEN)
        test_dataset = AppDataset(df_test, tokenizer, MAX_LEN)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        model = FraudDetector(n_meta_features=4) 
        model = model.to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.BCEWithLogitsLoss().to(DEVICE)

        print("🔥 学習を開始します...")
        
        for epoch in range(EPOCHS):
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)

            try:
                train_acc, train_loss = train_epoch(
                    model, train_loader, loss_fn, optimizer, None, len(df_train)
                )
                print(f'Train loss {train_loss} accuracy {train_acc}')
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("❌ GPUメモリ不足！BATCH_SIZEを下げてください。")
                    break
                else:
                    raise e

        # 1. 通常モデルの保存
        print("✅ 学習完了。オリジナルモデルを保存します。")
        torch.save(model.state_dict(), 'fraud_model.pth')
        
        # --- ★ここから量子化処理 (Cloud Run 2GB制限対応) ---
        print("\n📉 モデルを量子化（軽量化）しています...")
        
        # 量子化はCPU上で行う必要があるため、モデルをCPUへ移動
        model.to('cpu')
        model.eval()

        # 動的量子化の適用 (Linear層をint8に変換)
        # BERTのパラメータの大部分はLinear層なので、劇的に軽くなります
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear},  # 対象とする層
            dtype=torch.qint8
        )

        # ファイルサイズ比較用の出力
        org_size = os.path.getsize('fraud_model.pth') / 1024 / 1024
        print(f"📦 オリジナルサイズ: {org_size:.2f} MB")

        # 量子化モデルの保存
        torch.save(quantized_model.state_dict(), 'fraud_model_quantized.pth')
        
        q_size = os.path.getsize('fraud_model_quantized.pth') / 1024 / 1024
        print(f"💾 量子化モデル保存完了: fraud_model_quantized.pth ({q_size:.2f} MB)")
        print(f"🚀 圧縮率: {q_size/org_size*100:.1f}% (Cloud Run無料枠で動作可能です)")

    except FileNotFoundError:
        print("エラー: app_dataset_labeled.csv が見つかりません。")

if __name__ == "__main__":
    main()