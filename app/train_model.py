import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import gc  # ガベージコレクション用
from tqdm import tqdm # 進捗バー

# --- 設定 ---
BATCH_SIZE = 8       # GPUメモリ不足でエラーが出る場合はここを 8 や 4 に下げる
EPOCHS = 3            # 学習回数
LEARNING_RATE = 2e-5  # 学習率
MAX_LEN = 128         # 文章の最大長（長くするとメモリを食う）
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
        
        # 権限リストの特定（簡易版：実際はもっと多くの権限をリストアップする）
        self.risky_perms = [
            "SYSTEM_ALERT_WINDOW", "RECEIVE_BOOT_COMPLETED", 
            "BIND_ACCESSIBILITY_SERVICE", "READ_CONTACTS"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        # テキストデータの処理
        text = str(row['description']) if pd.notna(row['description']) else ""
        
        # 権限データの処理（One-hotエンコーディング的なもの）
        perms_str = str(row['permissions']) if pd.notna(row['permissions']) else ""
        perm_vec = [1.0 if rp in perms_str else 0.0 for rp in self.risky_perms]
        
        # BERT用トークナイズ
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
        
        # BERTの出力層直後にドロップアウトを入れる（過学習防止）
        self.drop = nn.Dropout(p=0.3)
        
        # メタデータ処理用の層
        self.meta_layer = nn.Linear(n_meta_features, 32)
        
        # 最終分類層 (BERTの768次元 + メタデータの32次元)
        self.out = nn.Linear(768 + 32, 1)

    def forward(self, input_ids, attention_mask, metadata):
        # BERTの処理
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # pooler_outputは[CLS]トークンの埋め込みベクトル
        bert_out = self.drop(outputs.pooler_output)
        
        # メタデータの処理
        meta_out = torch.relu(self.meta_layer(metadata))
        
        # 結合
        combined = torch.cat((bert_out, meta_out), dim=1)
        
        # 最終出力
        return self.out(combined)

# --- 3. 学習ループ関数 ---
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    # 混合精度学習のためのスケーラー
    scaler = torch.amp.GradScaler('cuda',enabled=torch.cuda.is_available())

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(DEVICE)
        attention_mask = d["attention_mask"].to(DEVICE)
        metadata = d["metadata"].to(DEVICE)
        labels = d["labels"].to(DEVICE)

        optimizer.zero_grad()

        # 混合精度コンテキスト（メモリ節約＆高速化）
        with torch.amp.autocast('cuda',enabled=torch.cuda.is_available()):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                metadata=metadata
            )
            # sigmoidで0~1にしてから損失計算したいが、
            # BCEWithLogitsLossを使うので生の出力(logit)を渡すのが安定的
            loss = loss_fn(outputs, labels.unsqueeze(1))

        # 誤差逆伝播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler:
            scheduler.step()

        preds = torch.sigmoid(outputs).round()
        correct_predictions += torch.sum(preds.flatten() == labels)
        losses.append(loss.item())
        
        # GPUメモリキャッシュを適宜クリア（おまじない）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return correct_predictions.double() / n_examples, np.mean(losses)

# --- メイン実行部 ---
def main():
    try:
        # データ読み込み
        print("📂 データを読み込んでいます...")
        df = pd.read_csv(r'C:\learn\scamwall\app_dataset_labeled.csv')
        
        # データ数が少なすぎるとエラーになるのでチェック
        if len(df) < 10:
            print("⚠️ データが少なすぎます。data_collector.pyでもっと集めてください。")
            return

        # 訓練用とテスト用に分割 (8:2)
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        
        tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
        
        # DataLoader作成
        train_dataset = AppDataset(df_train, tokenizer, MAX_LEN)
        test_dataset = AppDataset(df_test, tokenizer, MAX_LEN)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE) # テスト時はシャッフル不要

        # モデル初期化
        # 権限特徴量の数は AppDataset.risky_perms の長さと同じにする
        model = FraudDetector(n_meta_features=4) 
        model = model.to(DEVICE)

        # オプティマイザ設定
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.BCEWithLogitsLoss().to(DEVICE) # 2値分類用ロス関数

        print("🔥 学習を開始します...")
        
        for epoch in range(EPOCHS):
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)

            try:
                train_acc, train_loss = train_epoch(
                    model,
                    train_loader,
                    loss_fn,
                    optimizer,
                    None, # Schedulerは今回は省略
                    len(df_train)
                )
                print(f'Train loss {train_loss} accuracy {train_acc}')
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("❌ GPUメモリ不足が発生しました！")
                    print("対策: BATCH_SIZE を小さくしてください（現在: {}）".format(BATCH_SIZE))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                else:
                    raise e

        print("✅ 学習完了。モデルを保存します。")
        torch.save(model.state_dict(), 'fraud_model.pth')
        print("💾 fraud_model.pth として保存されました。")

    except FileNotFoundError:
        print("エラー: app_dataset_labeled.csv が見つかりません。labeler.py を先に実行してください。")

if __name__ == "__main__":
    main()