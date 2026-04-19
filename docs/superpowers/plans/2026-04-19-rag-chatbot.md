# RAG チャットボット実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 既存の社内FAQチャットボットを、ChromaDB + sentence-transformers によるRAGアプリに進化させる。

**Architecture:** `rag.py` に `RAGEngine` クラスを新規作成し、ChromaDB in-memory コレクションと `paraphrase-multilingual-MiniLM-L12-v2` モデルによるベクトル検索を実装する。`app.py` は RAGEngine を使うように改修し、複数PDF対応UIを追加する。

**Tech Stack:** Flask, OpenAI GPT-4o, ChromaDB (in-memory), sentence-transformers, PyPDF2

---

## ファイルマップ

| ファイル | 種別 | 役割 |
|---|---|---|
| `rag.py` | 新規 | RAGEngine クラス（ChromaDB・埋め込み・チャンク・検索） |
| `app.py` | 修正 | Flask ルーティング、RAGEngine 統合、複数PDF対応UI |
| `requirements.txt` | 修正 | chromadb, sentence-transformers 追加 |
| `tests/test_rag.py` | 新規 | RAGEngine の単体テスト |

---

### Task 1: 依存パッケージ追加

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: requirements.txt を更新する**

```
flask
openai
python-dotenv
PyPDF2
reportlab
chromadb
sentence-transformers
```

- [ ] **Step 2: パッケージをインストールする**

```bash
pip install -r requirements.txt
```

期待: エラーなくインストール完了

- [ ] **Step 3: コミット**

```bash
git add requirements.txt
git commit -m "feat: add chromadb and sentence-transformers dependencies"
```

---

### Task 2: RAGEngine の基盤（初期化・チャンク分割）

**Files:**
- Create: `rag.py`
- Create: `tests/test_rag.py`

- [ ] **Step 1: テストを書く（初期化とチャンク分割）**

`tests/test_rag.py` を作成する:

```python
import pytest
from rag import RAGEngine

@pytest.fixture(scope="module")
def engine():
    return RAGEngine()

def test_init_creates_collection(engine):
    assert engine.collection is not None
    assert engine.collection.count() == 0

def test_chunk_text_basic(engine):
    text = "あ" * 1200
    chunks = engine._chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 3
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 500

def test_chunk_text_short(engine):
    text = "短いテキスト"
    chunks = engine._chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == "短いテキスト"

def test_chunk_text_overlap(engine):
    text = "A" * 600
    chunks = engine._chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 2
    assert chunks[1][:100] == chunks[0][400:]
```

- [ ] **Step 2: テストが失敗することを確認する**

```bash
pytest tests/test_rag.py -v
```

期待: `ImportError: No module named 'rag'`

- [ ] **Step 3: rag.py を実装する**

```python
import re
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class RAGEngine:
    def __init__(self):
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._client = chromadb.EphemeralClient()
        self.collection = self._client.create_collection(
            name="faq_rag",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self.uploaded_pdfs = []

    def _chunk_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def _safe_id(s):
        return re.sub(r"[^a-zA-Z0-9_-]", "_", s)
```

- [ ] **Step 4: テストが通ることを確認する**

```bash
pytest tests/test_rag.py -v
```

期待: 4 passed（初回はモデルダウンロードで数分かかる場合あり）

- [ ] **Step 5: コミット**

```bash
git add rag.py tests/test_rag.py
git commit -m "feat: add RAGEngine base class with chunking"
```

---

### Task 3: add_faq() の実装

**Files:**
- Modify: `rag.py`
- Modify: `tests/test_rag.py`

- [ ] **Step 1: テストを追加する**

`tests/test_rag.py` の末尾に追加:

```python
def test_add_faq_stores_documents(engine):
    faqs = [
        {"q": "有給休暇の申請方法は？", "a": "マイページから申請してください。"},
        {"q": "経費精算の締め日は？", "a": "毎月末日です。"},
    ]
    engine.add_faq(faqs)
    assert engine.collection.count() == 2

def test_add_faq_replaces_existing(engine):
    faqs = [{"q": "新しい質問", "a": "新しい回答"}]
    engine.add_faq(faqs)
    assert engine.collection.count() == 1
```

- [ ] **Step 2: テストが失敗することを確認する**

```bash
pytest tests/test_rag.py::test_add_faq_stores_documents -v
```

期待: `AttributeError: 'RAGEngine' object has no attribute 'add_faq'`

- [ ] **Step 3: add_faq() を rag.py に追加する**

`RAGEngine` クラスに以下のメソッドを追加:

```python
    def add_faq(self, faqs):
        existing = self.collection.get(where={"type": "faq"})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
        if not faqs:
            return
        documents = [f"Q: {f['q']}\nA: {f['a']}" for f in faqs]
        ids = [f"faq_{i}" for i in range(len(faqs))]
        metadatas = [{"source": "faq", "type": "faq"} for _ in faqs]
        self.collection.add(documents=documents, ids=ids, metadatas=metadatas)
```

- [ ] **Step 4: テストが通ることを確認する**

```bash
pytest tests/test_rag.py -v
```

期待: 6 passed

- [ ] **Step 5: コミット**

```bash
git add rag.py tests/test_rag.py
git commit -m "feat: implement add_faq with replacement logic"
```

---

### Task 4: add_pdf() と remove_source() の実装

**Files:**
- Modify: `rag.py`
- Modify: `tests/test_rag.py`

- [ ] **Step 1: テストを追加する**

`tests/test_rag.py` の末尾に追加:

```python
def test_add_pdf_chunks_and_stores(engine):
    text = "社内規則の内容です。" * 100
    engine.add_pdf(text, "rules.pdf")
    assert "rules.pdf" in engine.uploaded_pdfs

def test_add_pdf_replaces_same_file(engine):
    count_before = engine.collection.count()
    text = "更新されたルールです。" * 10
    engine.add_pdf(text, "rules.pdf")
    count_after = engine.collection.count()
    assert count_after <= count_before

def test_remove_source(engine):
    engine.add_pdf("テスト文書。" * 20, "temp.pdf")
    engine.remove_source("temp.pdf")
    assert "temp.pdf" not in engine.uploaded_pdfs
    result = engine.collection.get(where={"source": "temp.pdf"})
    assert result["ids"] == []
```

- [ ] **Step 2: テストが失敗することを確認する**

```bash
pytest tests/test_rag.py::test_add_pdf_chunks_and_stores -v
```

期待: `AttributeError: 'RAGEngine' object has no attribute 'add_pdf'`

- [ ] **Step 3: add_pdf() と remove_source() を rag.py に追加する**

`RAGEngine` クラスに以下のメソッドを追加:

```python
    def add_pdf(self, text, filename):
        self.remove_source(filename)
        chunks = self._chunk_text(text)
        if not chunks:
            return
        safe = self._safe_id(filename)
        ids = [f"{safe}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "type": "pdf"} for _ in chunks]
        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        if filename not in self.uploaded_pdfs:
            self.uploaded_pdfs.append(filename)

    def remove_source(self, source):
        existing = self.collection.get(where={"source": source})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
        if source in self.uploaded_pdfs:
            self.uploaded_pdfs.remove(source)
```

- [ ] **Step 4: テストが通ることを確認する**

```bash
pytest tests/test_rag.py -v
```

期待: 9 passed

- [ ] **Step 5: コミット**

```bash
git add rag.py tests/test_rag.py
git commit -m "feat: implement add_pdf with chunking and remove_source"
```

---

### Task 5: search() の実装

**Files:**
- Modify: `rag.py`
- Modify: `tests/test_rag.py`

- [ ] **Step 1: テストを追加する**

`tests/test_rag.py` の末尾に追加:

```python
def test_search_returns_relevant_chunks(engine):
    faqs = [{"q": "有給休暇の申請方法は？", "a": "マイページから申請してください。"}]
    engine.add_faq(faqs)
    results = engine.search("有給休暇を取りたい", top_k=1)
    assert len(results) == 1
    assert "有給" in results[0]

def test_search_empty_collection_returns_empty():
    fresh = RAGEngine()
    results = fresh.search("何か質問")
    assert results == []

def test_search_top_k_limit(engine):
    results = engine.search("申請", top_k=2)
    assert len(results) <= 2
```

- [ ] **Step 2: テストが失敗することを確認する**

```bash
pytest tests/test_rag.py::test_search_returns_relevant_chunks -v
```

期待: `AttributeError: 'RAGEngine' object has no attribute 'search'`

- [ ] **Step 3: search() を rag.py に追加する**

`RAGEngine` クラスに以下のメソッドを追加:

```python
    def search(self, query, top_k=5):
        count = self.collection.count()
        if count == 0:
            return []
        n = min(top_k, count)
        results = self.collection.query(query_texts=[query], n_results=n)
        return results["documents"][0]
```

- [ ] **Step 4: テストが通ることを確認する**

```bash
pytest tests/test_rag.py -v
```

期待: 12 passed

- [ ] **Step 5: コミット**

```bash
git add rag.py tests/test_rag.py
git commit -m "feat: implement vector search with empty collection guard"
```

---

### Task 6: app.py を RAGEngine に統合する

**Files:**
- Modify: `app.py`

- [ ] **Step 1: app.py のインポートと初期化部分を更新する**

`app.py` の先頭（`from flask import ...` から `pdf_text = ""` まで）を以下に置き換える:

```python
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
from dotenv import load_dotenv
import os
import io
import PyPDF2
from rag import RAGEngine

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
rag = RAGEngine()

faq_data = [
    {"q": "有給休暇の申請方法は？", "a": "マイページから「休暇申請」をクリックして申請してください。"},
    {"q": "経費精算の締め日はいつですか？", "a": "毎月末日が締め日です。翌月10日までに申請してください。"},
    {"q": "リモートワークの申請はどうすればいいですか？", "a": "上長に事前にSlackで連絡し、承認を得てから実施してください。"},
]

rag.add_faq(faq_data)
```

- [ ] **Step 2: /ask ルートを RAG に更新する**

`app.py` の `/ask` ルートを以下に置き換える:

```python
@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '')
    chunks = rag.search(question, top_k=5)
    context = "\n\n".join(chunks) if chunks else "（関連情報が見つかりませんでした）"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "あなたは親切な社内FAQアシスタントです。"
                "以下の参考情報をもとに、ユーザーの質問に回答してください。"
                "参考情報に答えがない場合は「申し訳ありません、その情報は持っていません。担当部署にお問い合わせください。」と答えてください。\n\n"
                f"【参考情報】\n{context}"
            )},
            {"role": "user", "content": question}
        ]
    )
    return jsonify({"answer": response.choices[0].message.content})
```

- [ ] **Step 3: /faq POST ルートを更新する**

`app.py` の `/faq` POST ルートを以下に置き換える:

```python
@app.route('/faq', methods=['POST'])
def save_faq():
    global faq_data
    faq_data = request.json
    rag.add_faq(faq_data)
    return jsonify({"status": "ok"})
```

- [ ] **Step 4: /upload-pdf ルートを更新する**

`app.py` の `/upload-pdf` ルートを以下に置き換える:

```python
@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"success": False, "message": "ファイルが選択されていません"})
    file = request.files['pdf']
    if not file.filename:
        return jsonify({"success": False, "message": "ファイル名が空です"})
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = "".join(page.extract_text() or "" for page in reader.pages)
        rag.add_pdf(text, file.filename)
        return jsonify({
            "success": True,
            "message": f"{file.filename} を読み込みました（{len(reader.pages)}ページ）",
            "pdfs": rag.uploaded_pdfs
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
```

- [ ] **Step 5: /pdfs ルートを追加する**

`app.py` の `if __name__ == '__main__':` の直前に追加:

```python
@app.route('/pdfs', methods=['GET'])
def list_pdfs():
    return jsonify(rag.uploaded_pdfs)
```

- [ ] **Step 6: コミット**

```bash
git add app.py
git commit -m "feat: integrate RAGEngine into Flask routes"
```

---

### Task 7: UI を複数PDF対応に更新する

**Files:**
- Modify: `app.py` (HTML 文字列内)

- [ ] **Step 1: PDF ビューのHTMLを更新する**

`app.py` の HTML 内の `<div id="pdf-view">` セクションを以下に置き換える:

```html
  <div id="pdf-view">
    <div class="pdf-card">
      <h2>📄 PDFをアップロード</h2>
      <p>複数のPDFを読み込めます。アップロードした内容をベクトル検索してAIが回答します。</p>
      <div class="upload-area" onclick="document.getElementById('pdf-input').click()">
        <input type="file" id="pdf-input" accept=".pdf" multiple onchange="uploadPdfs(this)">
        <div class="icon">📂</div>
        <div class="text">クリックしてPDFを選択（複数可）</div>
        <div class="hint">または PDFファイルをここにドロップ</div>
      </div>
      <div id="status-box" class="status-box"></div>
      <div id="pdf-list" style="margin-top:16px;"></div>
    </div>
  </div>
```

- [ ] **Step 2: JavaScript の uploadPdf 関数を uploadPdfs に置き換える**

HTML の `<script>` ブロック内の `async function uploadPdf(input) { ... }` を以下に置き換える。
ファイル名は DOM API（textContent）で挿入してXSSを防止している:

```javascript
async function uploadPdfs(input) {
  const files = Array.from(input.files);
  if (!files.length) return;
  const status = document.getElementById('status-box');
  status.style.display = 'block';
  status.className = 'status-box';
  status.textContent = files.length + '件アップロード中...';
  let lastPdfs = [];
  for (const file of files) {
    const formData = new FormData();
    formData.append('pdf', file);
    const res = await fetch('/upload-pdf', { method: 'POST', body: formData });
    const data = await res.json();
    if (!data.success) {
      status.className = 'status-box status-error';
      status.textContent = file.name + ': ' + data.message;
      return;
    }
    lastPdfs = data.pdfs || [];
  }
  status.className = 'status-box status-success';
  status.textContent = files.length + '件のPDFを登録しました';
  renderPdfList(lastPdfs);
  input.value = '';
}

function renderPdfList(pdfs) {
  const el = document.getElementById('pdf-list');
  while (el.firstChild) el.removeChild(el.firstChild);
  if (!pdfs.length) return;
  const header = document.createElement('div');
  header.style.cssText = 'font-size:13px;color:#555;font-weight:600;margin-bottom:8px;';
  header.textContent = '登録済みPDF';
  el.appendChild(header);
  pdfs.forEach(function(name) {
    const item = document.createElement('div');
    item.style.cssText = 'padding:6px 12px;background:#f0f4ff;border-radius:8px;margin-bottom:4px;font-size:13px;';
    item.textContent = name;
    el.appendChild(item);
  });
}

async function loadPdfList() {
  const res = await fetch('/pdfs');
  const pdfs = await res.json();
  renderPdfList(pdfs);
}
```

- [ ] **Step 3: showTab を更新してPDFタブで一覧を読み込む**

`showTab` 関数を以下に置き換える:

```javascript
async function showTab(tab) {
  ['chat','admin','pdf'].forEach(t => {
    document.getElementById(t+'-view').style.display = t===tab?'block':'none';
    document.getElementById('btn-'+t).className = t===tab?'active':'';
  });
  if (tab === 'admin') await loadFaq();
  if (tab === 'pdf') await loadPdfList();
}
```

- [ ] **Step 4: コミット**

```bash
git add app.py
git commit -m "feat: update UI for multiple PDF upload with XSS-safe list display"
```

---

### Task 8: 動作確認

**Files:** なし（手動テスト）

- [ ] **Step 1: 全テストを実行する**

```bash
pytest tests/test_rag.py -v
```

期待: 12 passed

- [ ] **Step 2: アプリを起動する**

```bash
python app.py
```

期待: `Running on http://0.0.0.0:5000`（モデルロードに20〜60秒かかる場合あり）

- [ ] **Step 3: チャット動作を確認する**

ブラウザで `http://localhost:5000` を開き以下を確認する:
- 「有給休暇を取りたいんですが」→ FAQの回答が返ること
- 「経費の締め日を教えて」→ FAQの回答が返ること

- [ ] **Step 4: PDFアップロードを確認する**

PDFタブを開き、test.pdf をアップロードする。登録済みPDFリストに表示されることを確認する。

- [ ] **Step 5: 最終コミット**

```bash
git add -A
git commit -m "feat: complete RAG chatbot implementation"
```

---

### Task 9: GitHub にプッシュする

- [ ] **Step 1: プッシュする**

```bash
git push origin main
```

期待: `main -> main` が表示されてプッシュ完了
