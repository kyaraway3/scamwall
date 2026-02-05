## 📅 2026年2月5日時点：学習及び作成中
Note: 本プロジェクトは現在開発および学習のフェーズにあります。機能の追加やドキュメントの更新が頻繁に行われる可能性があります
[2026/02/05] BERT-base-japanese-v3 によるモデル構築完了

[2026/02/05] 学習成功（Accuracy: 83.8%）Loss（誤差）0.34

[Now] AndroidクライアントとのAPI連携実装中


## 📱 AI-Powered Fraud App Detection System

Google Playストアでのアプリインストール直前に、AIが詐欺や不審な挙動の兆候を検知し、ユーザーを守るAndroidセキュリティソリューションです。

# 📝 概要
本プロジェクトは、近年蔓延している「クリーナーアプリ」を装った広告詐欺や、過剰な権限要求を行う不審なアプリから一般ユーザーを保護するために開発されました。Accessibility Serviceを活用してPlayストアの情報をリアルタイムに抽出し、バックエンドのMLモデル（NLP + 特徴量分析）によってリスクをスコアリングします。

# ✨ 主な機能

リアルタイム・スキャン: Playストアのアプリ詳細画面を検知し、インストール前に自動解析。ハイブリッド・リスクスコアリング:NLP解析: アプリ説明文やユーザーレビューから、詐欺特有の言い回しやサクラレビューの傾向を抽出。
権限不一致検出: アプリのカテゴリ（例：電卓）に対し、不自然な権限要求（例：連絡先へのアクセス）を検知。
警告オーバーレイ: 危険度が高い場合、WindowManagerを使用してPlayストア上に警告メッセージを表示。
外部API連携: VirusTotal APIを用いた既知の脅威情報の統合。

🏗 システムアーキテクチャコンポーネント役割Android ClientKotlin / Accessibility Serviceによるデータ抽出 & 警告UI表示Backend APIFastAPI (Python) による推論エンドポイントの提供ML EnginePyTorch & Scikit-learnによるNLP解析およびリスク分類Database既知のシグネチャ、ホワイトリスト、解析ログの管理

🧠 機械学習・データ分析単なるルールベースではなく、自然言語処理と統計的特徴量を組み合わせた多角的な判定を行っています。分析アプローチ自然言語処理 (NLP): BERTモデルを用いて、説明文の「煽り」やレビュー内の不穏なキーワードをベクトル化。静的特徴量分析: 権限リスト、APKの難読化傾向、広告SDKの含有状況をLightGBM等のモデルで二値分類。

ラベリング基準: VirusTotalのアドウェア判定結果や、特定のキーワード（例：「広告が消えない」）を含むレビューを学習データとして活用。🛠 技術スタックAndroid: Kotlin, Coroutines, AccessibilityService, WindowManagerBackend: Python 3.x, FastAPI, DockerML/DS: PyTorch (BERT), Scikit-learn (LightGBM), PandasData Scraping: Google-play-scraper, BeautifulSoupSecurity: VirusTotal API

🚧 考慮事項Privacy First: ユーザー情報の匿名化およびプライバシーポリシーの策定。Low Latency: インストールボタンを押す前に判定を完了させるため、軽量なモデル設計と高速なAPIレスポンスを追求。

Compliance: Google Playの「アクセシビリティ権限」に関する最新ポリシーを遵守した設計