import os
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertJapaneseTokenizer, BertModel
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- Gemini設定 ---
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GENAI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- BERTモデル定義 ---
MODEL_NAME = 'cl-tohoku/bert-base-japanese-v3'

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

# --- サーバー起動時の準備 ---
app = FastAPI()
device = torch.device("cpu") # Cloud RunはCPU
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

# ★★★ モデルロード部分の修正 (量子化対応) ★★★
print("🔄 モデルを初期化中...")
n_perms = 4 
model = FraudDetector(n_meta_features=n_perms)

# 重要: 重みをロードする「前」に、モデル構造を量子化モードに変換する
# これを行わないと、保存されたint8の重みを読み込めません
model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear},  # BERTのLinear層を対象にする
    dtype=torch.qint8
)

# 量子化された重みファイルをロード
model_path = "fraud_model_quantized.pth"

if os.path.exists(model_path):
    print(f"📂 軽量化モデル {model_path} をロードしています...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ ロード成功！メモリ節約モードで稼働します。")
    except Exception as e:
        print(f"❌ ロードエラー: {e}")
        print("   train_model.py で正しく量子化モデルが保存されているか確認してください。")
else:
    print(f"⚠️ {model_path} が見つかりません！")
    print("   先に train_model.py を実行してモデルを作成してください。")

model.eval()

# リクエストの形式
class AppInfo(BaseModel):
    description: str
    permissions: list

# --- 判定ロジック ---
async def call_gemini_analysis(text, perms):
    prompt = f"""
    あなたはAndroidアプリのセキュリティ専門家です。
    以下のアプリ情報が「ユーザーを騙す詐欺アプリ」かどうかを判定し、
    理由とリスクレベル（0-1.0）を返してください。

    【アプリ説明文】: {text}
    【要求されている権限】: {', '.join(perms)}

    出力形式は必ず以下のJSONにしてください（Markdownなどの装飾は不要です）:
    {{"risk_score": 0.8, "reason": "ここに短い理由"}}
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f'{{"risk_score": 0.5, "reason": "Gemini API Error: {str(e)}"}}'

@app.post("/predict")
async def predict(info: AppInfo):
    # 1. BERTによる1次判定
    risky_perms = ["SYSTEM_ALERT_WINDOW", "RECEIVE_BOOT_COMPLETED", "BIND_ACCESSIBILITY_SERVICE", "READ_CONTACTS"]
    perm_vec = [1.0 if p in info.permissions else 0.0 for p in risky_perms]
    
    encoding = tokenizer(
        info.description,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        output = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            metadata=torch.tensor([perm_vec], dtype=torch.float)
        )
        bert_score = torch.sigmoid(output).item()

    # 2. ハイブリッド判定
    # スコアが 0.45 ~ 0.55 の微妙なラインの場合のみ Gemini に聞く
    if 0.45 <= bert_score <= 0.55:
        print(f"🤔 BERT迷い中(Score: {bert_score:.4f})... Geminiに相談します。")
        gemini_result = await call_gemini_analysis(info.description, info.permissions)
        return {
            "method": "Gemini (Hybrid)",
            "bert_raw_score": bert_score,
            "gemini_analysis": gemini_result
        }
    else:
        # BERTで即決
        print(f"⚡ BERT即決(Score: {bert_score:.4f})")
        return {
            "method": "BERT (Fast)",
            "score": bert_score,
            "is_fraud": bert_score > 0.5
        }