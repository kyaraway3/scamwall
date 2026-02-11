FROM python:3.10-slim

# エラーログを即座に出す設定
ENV PYTHONUNBUFFERED=True

WORKDIR /app

# 1. pipを最新にする（インストールの失敗を防ぐ）
RUN pip install --upgrade pip

# 2. CPU版のPyTorchを入れる
# transformers 4.46.3 はこのバージョンで快適に動きます
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. ライブラリを一括インストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. プログラムとモデルをコピー
COPY . .

# 5. 起動コマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]