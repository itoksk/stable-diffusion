# Stable Diffusion 画像生成プロジェクト

このリポジトリには、Stable Diffusionを使用した画像生成プログラムが含まれています。

## 📁 ファイル構成

- `version1.ipynb` - 基本的なStable Diffusion実装（Anything V5使用）
- `version2.py` - 複数モデル選択可能な改良版（ローカル実行用）
- `version2_colab.ipynb` - Google Colab用の複数モデル選択版

## 🎨 Version 2 の特徴

### 利用可能なモデル（すべて無料・安全）

#### アニメ・イラスト系
1. **Anything V5**
   - 万能型アニメモデル。幅広いアニメスタイルに対応
   - 初心者向け、汎用的なアニメイラスト

2. **Counterfeit V3.0**
   - 高品質なアニメイラスト特化
   - 繊細で美しいキャラクター表現、幻想的な雰囲気

#### 写実的・リアル系
3. **Realistic Vision V6.0**
   - 最高峰の写実的モデル
   - ポートレート、野生動物、都市風景

4. **EpicRealism**
   - 自然な肌の質感と照明
   - 人物写真、ファッション撮影

#### 万能・アート系
5. **DreamShaper**
   - リアルからファンタジーまで幅広く対応
   - SF・サイバーパンク風景、幻想的な世界観

## 🚀 使い方

### ローカル実行（version2.py）
```bash
# 必要なライブラリをインストール
pip install --upgrade diffusers[torch] transformers accelerate

# プログラムを実行
python version2.py
```

### Google Colab実行（version2_colab.ipynb）
1. Google Colabで`version2_colab.ipynb`を開く
2. ランタイムをGPU設定に変更
3. セルを順番に実行

## ⚖️ AI画像処理と著作権に関する重要な注意事項

### 2025年7月8日更新

#### AI生成画像の取り扱いについて

**絶対に避けるべき行為：**
- ❌ AI生成であることを伏せてSNS等に投稿
  - → 権利侵害による損害賠償請求の可能性
  - → 業務妨害罪で逮捕される可能性
- ❌ センシティブ画像の生成
  - → ハラスメントで停学処分などの可能性
  - → Google上でポルノ画像を生成した場合、自動検閲によりアカウントBAN処分

**重要な警告：**
- 教育機関で使用する場合、Colaboratoryファイル（プロンプト入力履歴含む）は教員も閲覧可能
- プライベートで画像生成を楽しむ場合は、個人のGoogleアカウントを使用すること

### プロンプトエンジニアリングのポイント

1. **効果的なプロンプト作成**
   - AIが理解しやすい具体的な条件を記述
   - 学習モデルの特徴を考慮
   - 英語で明確に記述（文の区切りは半角カンマ）

2. **Seed値の活用**
   - -1：ランダム生成（ガチャ要素）
   - 固定値：似た特徴の画像を再現可能
   - 気に入った画像のseed値をメモしておくと便利

### 技術的な制限事項

**Google Colab無料版の制限：**
- 連続稼働時間：約90分
- 制限超過後：約24時間の待機が必要
- 保存していないデータは消失するため注意

**推奨設定：**
- 画像サイズ：512x512（デフォルト）、最大768x768
- 生成時間の目安：512x512画像1枚あたり約10秒

## 📝 モデルごとの推奨プロンプト

### アニメ系モデル
```
基本構造: masterpiece, best quality, 1girl/1boy, [特徴], [服装], [背景], [ポーズ/表情]
例: masterpiece, best quality, 1girl, long hair, school uniform, classroom, smile
```

### リアル系モデル
```
基本構造: [撮影スタイル], [被写体], [照明], [構図], [カメラ設定]
例: professional portrait photography, young woman, natural lighting, close-up, bokeh background
```

### ネガティブプロンプト
- アニメ系：`nsfw`
- リアル系：`nsfw, low quality, blurry, cartoon, anime, painting, illustration`

## 🔧 トラブルシューティング

- セッション関係のエラー：「セッションの管理」から使用中のColaboratoryをすべて終了
- GPU設定：「ランタイム」→「ランタイムのタイプを変更」→「GPU」を選択

## 📄 ライセンス

すべての使用モデルは無料で各モデルのライセンスに準拠して利用してください。

## ⚠️ 免責事項

本プログラムを使用して生成された画像の利用については、使用者の責任において行ってください。不適切な使用による問題について、開発者は一切の責任を負いません。