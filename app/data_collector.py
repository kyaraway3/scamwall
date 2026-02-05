import pandas as pd
import time
import random
from google_play_scraper import search, app, Sort
from tqdm import tqdm

class PlayStoreScraper:
    def __init__(self, lang='ja', country='jp'):
        self.lang = lang
        self.country = country
        self.apps_data = []

    def search_app_ids(self, queries, limit=50):
        """
        キーワード検索でアプリID（パッケージ名）のリストを収集する
        """
        app_ids = set() # 重複を防ぐためにsetを使用
        print(f"🔍 検索を開始します: {queries}")

        for query in queries:
            try:
                results = search(
                    query,
                    lang=self.lang,
                    country=self.country,
                    n_hits=limit # 各クエリで何件取得するか
                )
                for res in results:
                    app_ids.add(res['appId'])
                
                print(f"  - '{query}' で {len(results)} 件取得")
                time.sleep(random.uniform(1, 3)) # サーバー負荷軽減のための待機
            except Exception as e:
                print(f"Error searching {query}: {e}")
        
        print(f"✅ 合計 {len(app_ids)} 個のユニークなアプリIDを収集しました。")
        return list(app_ids)

    def fetch_details(self, app_ids):
        """
        アプリIDのリストから詳細情報（説明文、権限など）を取得する
        """
        print("📥 詳細情報の取得を開始します...")
        
        for app_id in tqdm(app_ids): # 進捗バーを表示
            try:
                # APIリクエスト
                details = app(
                    app_id,
                    lang=self.lang,
                    country=self.country
                )
                
                # 必要なデータだけを抽出して辞書にする
                extracted_data = {
                    'app_id': details.get('appId'),
                    'title': details.get('title'),
                    'description': details.get('description'), # BERT用重要データ
                    'summary': details.get('summary'),
                    'score': details.get('score'),
                    'ratings': details.get('ratings'),
                    'reviews': details.get('reviews'), # レビュー数
                    'installs': details.get('installs'),
                    'developer': details.get('developer'),
                    'developer_email': details.get('developerEmail'),
                    'developer_website': details.get('developerWebsite'),
                    'updated': details.get('updated'),
                    'contains_ads': details.get('adSupported'), # 広告の有無
                    # 権限リストはAI学習用に文字列化して保存
                    'permissions': ",".join([p['permission'] for p in details.get('permissions') or []]),
                    'icon_url': details.get('icon')
                }
                
                self.apps_data.append(extracted_data)
                
                # スクレイピング検知回避のためランダムに待機
                time.sleep(random.uniform(0.5, 1.5))

            except Exception as e:
                # 削除されたアプリなどでエラーが出ても止まらないようにする
                # print(f"Error fetching {app_id}: {e}")
                continue

    def save_to_csv(self, filename='dataset.csv'):
        """データをCSVに保存"""
        if not self.apps_data:
            print("保存するデータがありません。")
            return
            
        df = pd.DataFrame(self.apps_data)
        # ラベル列（is_fraud）は後で人間またはルールベースで埋めるため空けておく、または仮置き
        if 'is_fraud' not in df.columns:
            df['is_fraud'] = -1 # -1: 未判定
            
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"💾 データを {filename} に保存しました。 ({len(df)}件)")

# --- 実行部 ---
if __name__ == "__main__":
    scraper = PlayStoreScraper()
    
    # 1. 詐欺アプリが多そうなキーワード（ターゲット）
    risky_keywords = [
        "phone cleaner", "battery booster", "cpu cooler", 
        "ram booster", "virus cleaner", "free antivirus",
        "スマホ最適化", "バッテリー長持ち"
    ]
    
    # 2. 正常なアプリも混ぜるためのキーワード（比較対象）
    safe_keywords = [
        "browser", "camera", "clock", "calculator", "SNS", "news"
    ]
    
    # ID収集
    target_ids = scraper.search_app_ids(risky_keywords, limit=30)
    safe_ids = scraper.search_app_ids(safe_keywords, limit=10) # バランス調整
    
    all_ids = target_ids + safe_ids
    
    # 詳細取得
    scraper.fetch_details(all_ids)
    
    # CSV保存
    scraper.save_to_csv("app_dataset_raw.csv")