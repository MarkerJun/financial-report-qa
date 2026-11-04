import os
from flask import Flask,request,jsonify,send_file
from engine import Engine,ROOT
app=Flask(__name__); engine=Engine()
@app.get('/')
def home(): return send_file(ROOT/'index.html')
@app.get('/api/status')
def status(): return jsonify(companies=engine.companies,chunks=len(engine.chunks),model_configured=bool(os.getenv('LLM_API_KEY')))
@app.post('/api/ask')
def ask():
    data=request.get_json(); question=str(data.get('question','')).strip()
    if not question or len(question)>1000: return jsonify(error='请输入1至1000字的问题'),400
    try: return jsonify(engine.ask(question,data.get('mode','hybrid')))
    except Exception as e: return jsonify(error=str(e)),500
@app.get('/pdf/<code>')
def pdf(code):
    if code not in [c['code'] for c in engine.chunks]: return '',404
    return send_file(ROOT/f'data/pdf/{code}_2025.pdf')
if __name__=='__main__': app.run(host='127.0.0.1',port=8033,debug=False)
