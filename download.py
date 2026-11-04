"""从巨潮资讯下载完整年报；校验 PDF 文件头并保存 SHA256。"""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT = Path(__file__).resolve().parent

def download(item):
    path = ROOT / 'data' / 'pdf' / f"{item['code']}_{item['year']}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        session = requests.Session()
        session.trust_env = False
        response = session.get(item['url'], timeout=120)
        response.raise_for_status()
        if not response.content.startswith(b'%PDF'):
            raise ValueError(f"不是 PDF: {item['company']}")
        path.write_bytes(response.content)
    item = dict(item, file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    print(item['company'], item['bytes'], flush=True)
    return item

if __name__ == '__main__':
    sources = json.loads((ROOT / 'data/sources.json').read_text(encoding='utf-8'))
    results = list(ThreadPoolExecutor(max_workers=4).map(download, sources))
    (ROOT / 'data/manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
