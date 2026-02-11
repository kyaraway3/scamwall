import torch
import torch.nn as nn
from transformers import BertJapaneseTokenizer, BertModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv

# .env ファイルの内容を読み込む
load_dotenv()

# --- 設定 ---
# 変数として取得（第2引数はデフォルト値）
# デフォルト値を設定しない（.envに書いてないとエラーで止まるようにする）
MODEL_PATH = os.getenv('MODEL_PATH')

if MODEL_PATH is None:
    raise ValueError(".envファイルに MODEL_PATH が設定されていません！")
PORT = int(os.getenv('API_PORT', 8080))
HOST = os.getenv('API_HOST', '127.0.0.1')
MODEL_NAME = 'cl-tohoku/bert-base-japanese-v3'
MAX_LEN = 128

# デバイス設定（推論はCPUでも十分速いですが、GPUがあれば使う）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. モデル定義 (学習時と同じ構造にする必要があります) ---
class FraudDetector(nn.Module):
    def __init__(self, n_meta_features):
        super(FraudDetector, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.meta_layer = nn.Linear(n_meta_features, 32)
        self.out = nn.Linear(768 + 32, 1)

    def forward(self, input_ids, attention_mask, metadata):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_out = self.drop(outputs.pooler_output)
        meta_out = torch.relu(self.meta_layer(metadata))
        combined = torch.cat((bert_out, meta_out), dim=1)
        return self.out(combined)

# --- 2. サーバーの準備 ---
app = FastAPI()

# リクエストの受け取り型（JSONの形）
class AppRequest(BaseModel):
    description: str
    permissions: list[str]

# グローバル変数としてモデル等を保持
model = None
tokenizer = None
risky_perms_list = [
    "SYSTEM_ALERT_WINDOW", "RECEIVE_BOOT_COMPLETED", 
    "BIND_ACCESSIBILITY_SERVICE", "READ_CONTACTS"
]

@app.on_event("startup")
def load_model():
    global model, tokenizer
    print("🚀 サーバー起動中...モデルをロードしています...")
    
    # トークナイザーのロード
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
    
    # モデル構造の準備
    model = FraudDetector(n_meta_features=len(risky_perms_list))
    
    # 学習済み重みの読み込み
    try:
        # map_location='cpu' は、GPUで学習したものをCPUサーバーで動かす場合に必要（自動調整）
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval() # 推論モード（学習しない設定）へ
        print("✅ モデルロード完了！準備OKです。")
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        print(f"パスを確認してください: {MODEL_PATH}")

# --- 3. 判定APIのエンドポイント ---
@app.post("/analyze")
async def analyze_app(request: AppRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. テキストの前処理
    encoded = tokenizer(
        request.description,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # 2. 権限の前処理
    perm_vec = [1.0 if rp in request.permissions else 0.0 for rp in risky_perms_list]
    meta_tensor = torch.tensor([perm_vec], dtype=torch.float)

    # 3. GPU/CPUへ転送
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)
    meta_tensor = meta_tensor.to(DEVICE)

    # 4. 推論実行
    with torch.no_grad():
        output = model(input_ids, attention_mask, meta_tensor)
        prediction = torch.sigmoid(output).item() # 0.0 ~ 1.0 の確率に変換

    # 5. 結果を返す
    return {
        "risk_score": prediction, # 0.95なら95%詐欺
        "is_fraud": prediction > 0.5,
        "message": "危険なアプリの可能性があります" if prediction > 0.5 else "安全そうです"
    }

# Pythonから直接実行した場合の起動コマンド
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)