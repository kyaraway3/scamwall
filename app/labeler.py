import pandas as pd
import ast # 文字列化されたリストを元に戻すのに使います

def apply_labeling_rules(row):
    """
    1行（1アプリ）ごとのデータを読んで、詐欺かどうか判定するルール
    戻り値: 1 (詐欺疑い), 0 (安全そう)
    """
    # データが空の場合の対策
    desc = str(row['description']) if pd.notna(row['description']) else ""
    title = str(row['title']) if pd.notna(row['title']) else ""
    perms = str(row['permissions']) if pd.notna(row['permissions']) else ""
    dev_email = str(row['developer_email']) if pd.notna(row['developer_email']) else ""
    
    # --- 判定スコア計算 ---
    score = 0
    
    # 1. キーワード判定（煽り文句）
    # 実際にデータを見て、詐欺アプリによくある単語を追加していきます
    danger_keywords = [
        "CPUクーラー", "40GB節約", "今すぐ修復", "ウイルスが検出されました", 
        "バッテリーを冷やす", "RAMを増やす", "ブースト", "1タップで解決",
        "Booster", "Cleaner", "Optimizer", "Free", "Fast", "Speed", 
        "加速", "最適化", "掃除", "冷却", "無料"
    ]
    
    # タイトルや説明文に危険ワードが含まれているか
    hit_words = [word for word in danger_keywords if word in desc or word in title]
    if len(hit_words) > 0:
        score += 2 # 危険ワードがあれば+2点
        # print(f"Keyword Hit: {hit_words} in {row['app_id']}") # デバッグ用

    # 2. 権限判定（不審な権限）
    risky_perms = [
        "SYSTEM_ALERT_WINDOW", # 画面オーバーレイ
        "RECEIVE_BOOT_COMPLETED", # 自動起動
        "BIND_ACCESSIBILITY_SERVICE" # ユーザー補助（操作乗っ取り）
    ]
    if any(p in perms for p in risky_perms):
        score += 3 # 危険な権限があれば+3点

    # 3. 開発者判定
    # Gmailなどのフリーメールを使っている企業は怪しい
    free_domains = ["@gmail.com", "@yahoo.com", "@hotmail.com", "@outlook.com"]
    if any(domain in dev_email for domain in free_domains):
        score += 1

    # --- 最終判定 ---
    # スコアが一定以上なら「詐欺(1)」とする
    threshold = 3 
    return 1 if score >= threshold else 0

def main():
    input_file = 'app_dataset_raw.csv'
    output_file = 'app_dataset_labeled.csv'
    
    try:
        # CSVを読み込む
        print(f"📖 {input_file} を読み込んでいます...")
        df = pd.read_csv(input_file)
        
        # ラベル付け関数を適用
        # tqdmを使うと進捗が見えますが、今回は一瞬なので省略
        print("🏷️ ラベル付けを実行中...")
        df['is_fraud'] = df.apply(apply_labeling_rules, axis=1)
        
        # 結果の確認（詐欺判定された数を表示）
        fraud_count = df['is_fraud'].sum()
        total_count = len(df)
        print(f"📊 判定結果: 全 {total_count} 件中、 {fraud_count} 件を「詐欺疑い」と判定しました。")
        
        # 保存
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"💾 ラベル付きデータを {output_file} に保存しました。")
        
        # 詐欺判定されたアプリの名前を一部表示してみる（確認用）
        if fraud_count > 0:
            print("\n--- ⚠️ 詐欺判定されたアプリ例 ---")
            print(df[df['is_fraud'] == 1][['title', 'is_fraud']].head(5))

    except FileNotFoundError:
        print(f"エラー: {input_file} が見つかりません。先に data_collector.py を実行してください。")

if __name__ == "__main__":
    main()