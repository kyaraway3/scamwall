import requests
import json
import os
from dotenv import load_dotenv

# .envを読み込む
load_dotenv()

# URLを環境変数から取得（なければローカルを見るように設定）
url = os.getenv("API_URL", "http://127.0.0.1:8080/predict")

print(f"📡 接続先: {url}")

# --- 以下、テストデータ ---

data = {
    "description": "【緊急】お客様の口座が凍結されました。解除するには連絡先へのアクセスを許可し、以下のリンクから直ちに手続きを行ってください。さもなくば法的措置を取ります。",
    "permissions": ["READ_CONTACTS", "SEND_SMS", "RECEIVE_SMS"]
}

# 送信
try:
    response = requests.post(url, json=data)
    print(f"ステータスコード: {response.status_code}")
    print("判定結果:", json.dumps(response.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print(f"エラー: {e}")