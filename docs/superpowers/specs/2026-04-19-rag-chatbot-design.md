# RAG チャットボット設計書

**日付:** 2026-04-19  
**対象プロジェクト:** ai-app (社内FAQチャットボット → RAGアプリ進化)

---

## 概要

現在の社内FAQチャットボットを、ChromaDB + sentence-transformers を使ったRAG（Retrieval-Augmented Generation）アプリに進化させる。複数PDFを大量に読み込み可能にし、ベクトル検索で関連部分のみを抽出してAIに渡す仕組みを構築する。

---

## アーキテクチャ

### ファイル構成

```
ai-app/
├── app.py          # Flask ルーティング（変更）
├── rag.py          # RAGエンジン（新規）
├── requirements.txt  # 依存パッケージ追加
└── render.yaml     # 変更なし
```

### 構造図

```
app.py (Flask)
  ├── GET  /             → チャットUI（複数PDFアップロード対応）
  ├── POST /ask          → RAGEngine.search() → GPT-4o
  ├── POST /upload-pdf   → RAGEngine.add_pdf()
  ├── GET  /faq          → FAQリスト返却
  └── POST /faq          → FAQデータ更新 + RAGEngine再登録

rag.py (RAGEngine クラス)
  ├── __init__()         → ChromaDB in-memory + モデル初期化 + FAQ自動登録
  ├── add_faq(faqs)      → FAQをチャンク化してChromaDBに登録
  ├── add_pdf(text, filename) → PDFテキストをチャンク分割してChromaDBに登録
  ├── search(query, top_k=5) → ベクトル検索で関連チャンクを返す
  └── remove_source(source)  → 特定ソースのチャンクを削除（PDF再登録用）
```

---

## データフロー

### 質問応答フロー

1. ユーザーが質問を送信
2. `RAGEngine.search(question, top_k=5)` を呼び出し
3. ChromaDB がコサイン類似度でTop5チャンクを返す
4. チャンクをコンテキストとして GPT-4o に渡す（最大~2000文字相当）
5. GPT-4o が回答を生成して返す

### PDFアップロードフロー

1. ユーザーがPDFをアップロード（複数対応）
2. PyPDF2 でテキスト抽出
3. テキストを500文字・100文字オーバーラップでチャンク分割
4. `RAGEngine.add_pdf(text, filename)` で各チャンクをChromaDBに登録
5. チャンクIDは `{filename}_{index}` 形式で一意に管理

### 起動フロー

1. Flask 起動
2. `RAGEngine.__init__()` でChromaDB（in-memory）初期化
3. sentence-transformers モデル（`paraphrase-multilingual-MiniLM-L12-v2`）ロード
4. `add_faq(faq_data)` でFAQ3件を自動登録

---

## RAGエンジン詳細

### チャンク戦略

- **チャンクサイズ:** 500文字（日本語は1文字≒1トークンのため）
- **オーバーラップ:** 100文字（文脈の連続性を保持）
- **メタデータ:** `source`（ファイル名または"faq"）、`type`（"faq"/"pdf"）

### ベクトル埋め込み

- モデル: `paraphrase-multilingual-MiniLM-L12-v2`
- ChromaDB のカスタム埋め込み関数として実装
- コレクション名: `"faq_rag"`

### 検索

- `top_k=5` でTop5チャンクを取得
- 距離スコアが閾値（0.9）以上の場合は「関連情報なし」と判断
- FAQとPDFを区別せず統合検索

---

## UIの変更点

- PDFアップロードを複数ファイル対応に変更（`multiple` 属性）
- アップロード済みPDF一覧を表示するリストを追加
- FAQ保存後に自動でChromaDBを更新するロジックを追加

---

## エラーハンドリング

- PDF解析失敗時: エラーメッセージをJSONで返す
- ChromaDB登録失敗時: ログ出力してスキップ（チャットは継続）
- 検索結果が空の場合: AIに「関連情報が見つかりません」と伝えて対応

---

## 依存パッケージ追加

```
chromadb
sentence-transformers
```

---

## デプロイ考慮事項

- ChromaDB は **in-memory** のみ使用（Renderの無料プランに対応）
- アプリ再起動時: FAQは自動登録、PDFは再アップロードが必要
- sentence-transformers モデルは初回起動時にダウンロード（約470MB）
- Renderのビルド時間増加に注意（モデルキャッシュなし）
