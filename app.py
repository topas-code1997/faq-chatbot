from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
import PyPDF2
import io

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

faq_data = [
        {"q": "有給休暇の申請方法は？", "a": "マイページから「休暇申請」をクリックして申請してください。"},
    {"q": "経費精算の締め日はいつですか？", "a": "毎月末日が締め日です。翌月10日までに申請してください。"},
    {"q": "リモートワークの申請はどうすればいいですか？", "a": "上長に事前にSlackで連絡し、承認を得てから実施してください。"},
]

pdf_text = ""

HTML = """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>社内FAQチャットボット</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f7fb; min-height: 100vh; display: flex; flex-direction: column; }
  header { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 16px 32px; display: flex; align-items: center; gap: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.15); }
  header .logo { width: 36px; height: 36px; background: #4f8ef7; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; }
  header h1 { color: white; font-size: 18px; font-weight: 600; }
  header span { color: #8899bb; font-size: 12px; margin-left: 4px; }
  nav { background: white; border-bottom: 1px solid #e8ecf4; padding: 0 32px; display: flex; gap: 0; }
  nav button { padding: 14px 24px; border: none; background: none; cursor: pointer; font-size: 14px; font-weight: 500; color: #888; border-bottom: 2px solid transparent; transition: all 0.2s; }
  nav button.active { color: #4f8ef7; border-bottom-color: #4f8ef7; }
  nav button:hover { color: #4f8ef7; }
  .container { max-width: 800px; margin: 32px auto; width: 100%; padding: 0 20px; flex: 1; }
  #chat-view, #admin-view, #pdf-view { display: none; }
  .chat-box { background: white; border-radius: 16px; box-shadow: 0 2px 20px rgba(0,0,0,0.06); overflow: hidden; }
  #chat { height: 420px; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 12px; }
  .user-msg { display: flex; justify-content: flex-end; }
  .user-msg span { background: linear-gradient(135deg, #4f8ef7, #6c6ef7); color: white; padding: 10px 16px; border-radius: 18px 18px 4px 18px; max-width: 75%; font-size: 14px; line-height: 1.5; }
  .bot-msg { display: flex; justify-content: flex-start; align-items: flex-start; gap: 8px; }
  .bot-avatar { width: 32px; height: 32px; background: #eef2ff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }
  .bot-msg span { background: #f5f7fb; color: #333; padding: 10px 16px; border-radius: 18px 18px 18px 4px; max-width: 75%; font-size: 14px; line-height: 1.5; border: 1px solid #e8ecf4; }
  .input-area { padding: 16px 24px; border-top: 1px solid #e8ecf4; display: flex; gap: 10px; background: white; }
  .input-area input { flex: 1; padding: 12px 16px; border: 1px solid #e0e6f0; border-radius: 24px; font-size: 14px; outline: none; transition: border 0.2s; }
  .input-area input:focus { border-color: #4f8ef7; }
  .send-btn { padding: 12px 24px; background: linear-gradient(135deg, #4f8ef7, #6c6ef7); color: white; border: none; border-radius: 24px; cursor: pointer; font-size: 14px; font-weight: 500; transition: opacity 0.2s; }
  .send-btn:hover { opacity: 0.9; }
  .admin-card { background: white; border-radius: 16px; box-shadow: 0 2px 20px rgba(0,0,0,0.06); padding: 24px; }
  .admin-card h2 { font-size: 16px; color: #333; margin-bottom: 16px; font-weight: 600; }
  .faq-item { border: 1px solid #e8ecf4; border-radius: 12px; padding: 16px; margin-bottom: 12px; transition: box-shadow 0.2s; }
  .faq-item:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
  .faq-item-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .faq-item-header span { font-size: 12px; font-weight: 600; color: #4f8ef7; background: #eef2ff; padding: 4px 10px; border-radius: 20px; }
  .del-btn { background: #fff0f0; color: #ff4444; border: none; border-radius: 8px; padding: 6px 12px; cursor: pointer; font-size: 12px; font-weight: 500; }
  .faq-label { font-size: 12px; color: #888; margin-bottom: 4px; font-weight: 500; }
  .faq-item input, .faq-item textarea { width: 100%; padding: 10px 12px; border: 1px solid #e8ecf4; border-radius: 8px; font-size: 14px; outline: none; transition: border 0.2s; font-family: inherit; }
  .faq-item input:focus, .faq-item textarea:focus { border-color: #4f8ef7; }
  .faq-item textarea { height: 70px; resize: none; margin-top: 8px; }
  .faq-item input { margin-top: 4px; }
  .btn-row { display: flex; gap: 10px; margin-top: 16px; }
  .add-btn { flex: 1; padding: 12px; background: #f5f7fb; color: #4f8ef7; border: 1px dashed #4f8ef7; border-radius: 10px; cursor: pointer; font-size: 14px; font-weight: 500; }
  .save-btn { flex: 1; padding: 12px; background: linear-gradient(135deg, #4f8ef7, #6c6ef7); color: white; border: none; border-radius: 10px; cursor: pointer; font-size: 14px; font-weight: 500; }
  .typing { display: flex; gap: 4px; align-items: center; padding: 10px 16px; background: #f5f7fb; border-radius: 18px; border: 1px solid #e8ecf4; }
  .typing span { width: 6px; height: 6px; background: #aaa; border-radius: 50%; animation: bounce 1s infinite; }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-6px)} }
  .pdf-card { background: white; border-radius: 16px; box-shadow: 0 2px 20px rgba(0,0,0,0.06); padding: 24px; }
  .pdf-card h2 { font-size: 16px; color: #333; margin-bottom: 8px; font-weight: 600; }
  .pdf-card p { font-size: 13px; color: #888; margin-bottom: 20px; }
  .upload-area { border: 2px dashed #c0d0f0; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.2s; }
  .upload-area:hover { border-color: #4f8ef7; background: #f0f4ff; }
  .upload-area input { display: none; }
  .upload-area .icon { font-size: 40px; margin-bottom: 12px; }
  .upload-area .text { font-size: 14px; color: #666; }
  .upload-area .hint { font-size: 12px; color: #aaa; margin-top: 4px; }
  .status-box { margin-top: 16px; padding: 12px 16px; border-radius: 10px; font-size: 14px; display: none; }
  .status-success { background: #f0fff4; border: 1px solid #34c759; color: #1a7a3a; }
  .status-error { background: #fff0f0; border: 1px solid #ff4444; color: #cc0000; }
</style>
</head>
<body>
<header>
  <div class="logo">💬</div>
  <h1>社内FAQチャットボット <span>AI powered</span></h1>
</header>
<nav>
  <button id="btn-chat" class="active" onclick="showTab('chat')">💬 チャット</button>
  <button id="btn-admin" onclick="showTab('admin')">⚙️ FAQ管理</button>
  <button id="btn-pdf" onclick="showTab('pdf')">📄 PDF読み込み</button>
</nav>
<div class="container">
  <div id="chat-view">
    <div class="chat-box">
      <div id="chat">
        <div class="bot-msg"><div class="bot-avatar">🤖</div><span>こんにちは！社内FAQアシスタントです。FAQやアップロードしたPDFの内容について何でもご質問ください。</span></div>
      </div>
      <div class="input-area">
        <input type="text" id="input" placeholder="質問を入力してください..." onkeydown="if(event.key==='Enter')send()">
        <button class="send-btn" onclick="send()">送信</button>
      </div>
    </div>
  </div>
  <div id="admin-view">
    <div class="admin-card">
      <h2>FAQ一覧</h2>
      <div id="faq-list"></div>
      <div class="btn-row">
        <button class="add-btn" onclick="addFaq()">＋ FAQを追加</button>
        <button class="save-btn" onclick="saveFaq()">💾 保存する</button>
      </div>
    </div>
  </div>
  <div id="pdf-view">
    <div class="pdf-card">
      <h2>📄 PDFをアップロード</h2>
      <p>社内マニュアルや規則集をアップロードすると、その内容をもとに回答できるようになります。</p>
      <div class="upload-area" onclick="document.getElementById('pdf-input').click()">
        <input type="file" id="pdf-input" accept=".pdf" onchange="uploadPdf(this)">
        <div class="icon">📂</div>
        <div class="text">クリックしてPDFを選択</div>
        <div class="hint">または PDFファイルをここにドロップ</div>
      </div>
      <div id="status-box" class="status-box"></div>
    </div>
  </div>
</div>
<script>
let faqs = [];
async function showTab(tab) {
  ['chat','admin','pdf'].forEach(t => {
    document.getElementById(t+'-view').style.display = t===tab?'block':'none';
    document.getElementById('btn-'+t).className = t===tab?'active':'';
  });
  if(tab==='admin') await loadFaq();
}
async function loadFaq() {
  const res = await fetch('/faq');
  faqs = await res.json();
  renderFaq();
}
function renderFaq() {
  const list = document.getElementById('faq-list');
  list.innerHTML = faqs.map((f,i) => `
    <div class="faq-item">
      <div class="faq-item-header"><span>FAQ #${i+1}</span><button class="del-btn" onclick="deleteFaq(${i})">削除</button></div>
      <div class="faq-label">質問</div>
      <input type="text" value="${f.q}" onchange="faqs[${i}].q=this.value">
      <div class="faq-label">回答</div>
      <textarea onchange="faqs[${i}].a=this.value">${f.a}</textarea>
    </div>
  `).join('');
}
function addFaq() { faqs.push({q:'',a:''}); renderFaq(); }
function deleteFaq(i) { faqs.splice(i,1); renderFaq(); }
async function saveFaq() {
  await fetch('/faq',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(faqs)});
  alert('保存しました！');
}
async function uploadPdf(input) {
  const file = input.files[0];
  if(!file) return;
  const formData = new FormData();
  formData.append('pdf', file);
  const status = document.getElementById('status-box');
  status.style.display = 'block';
  status.className = 'status-box';
  status.textContent = '📤 アップロード中...';
  const res = await fetch('/upload-pdf', {method:'POST', body: formData});
  const data = await res.json();
  if(data.success) {
    status.className = 'status-box status-success';
    status.textContent = '✅ ' + data.message;
  } else {
    status.className = 'status-box status-error';
    status.textContent = '❌ エラーが発生しました';
  }
}
async function send() {
  const input = document.getElementById('input');
  const chat = document.getElementById('chat');
  const q = input.value.trim();
  if(!q) return;
  chat.innerHTML += `<div class="user-msg"><span>${q}</span></div>`;
  input.value = '';
  chat.innerHTML += `<div class="bot-msg" id="typing"><div class="bot-avatar">🤖</div><div class="typing"><span></span><span></span><span></span></div></div>`;
  chat.scrollTop = chat.scrollHeight;
  const res = await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const data = await res.json();
  document.getElementById('typing').remove();
  chat.innerHTML += `<div class="bot-msg"><div class="bot-avatar">🤖</div><span>${data.answer}</span></div>`;
  chat.scrollTop = chat.scrollHeight;
}
showTab('chat');
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/faq', methods=['GET'])
def get_faq():
    return jsonify(faq_data)

@app.route('/faq', methods=['POST'])
def save_faq():
    global faq_data
    faq_data = request.json
    return jsonify({"status": "ok"})

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    global pdf_text
    if 'pdf' not in request.files:
        return jsonify({"success": False})
    file = request.files['pdf']
    reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    pdf_text = text
    return jsonify({"success": True, "message": f"PDFを読み込みました（{len(reader.pages)}ページ）"})

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '')
    faq_text = "\n".join([f"Q: {f['q']}\nA: {f['a']}" for f in faq_data])
    context = f"【FAQ情報】\n{faq_text}"
    if pdf_text:
        context += f"\n\n【PDFドキュメント】\n{pdf_text[:3000]}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"あなたは親切な社内FAQアシスタントです。以下の情報をもとに、ユーザーの質問の意図を汲み取って柔軟に回答してください。情報にない場合は「申し訳ありません、その情報は持っていません。担当部署にお問い合わせください。」と答えてください。\n\n{context}"},
            {"role": "user", "content": question}
        ]
    )
    return jsonify({"answer": response.choices[0].message.content})

if __name__ == '__main__':
    app.run(debug=True)