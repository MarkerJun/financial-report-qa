"""按页抽取；表格按行列保存；正文切块保持公司、章节、页码。"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / '.tmp_pdf_deps'))
import pymupdf

def main():
    manifests=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    chunks=[]; pages=[]
    for source in manifests:
        doc=pymupdf.open(ROOT/source['file']); section='封面与目录'; count=0
        for number,page in enumerate(doc,1):
            text=page.get_text('text',sort=True)
            headings=re.findall(r'(?:第[一二三四五六七八九十]+节[^\n]{2,45}|[一二三四五六七八九十]+、[^\n]{2,40})',text)
            if headings: section=headings[0]
            printed=re.findall(r'^\s*(\d{1,3})(?:\s*/\s*\d+)?\s*$',text,re.M)
            base={k:source[k] for k in ['company','code','year','url']}
            base.update(page=number,printed_page=printed[-1] if printed else None,section=section)
            pages.append(dict(base,text=text))
            tables=[]
            # Only run table detection on pages containing financial/table signals.
            if re.search(r'单位[：:]|本期发生额|本报告期|2025\s*年.*2024',text):
                try:
                    for table in page.find_tables().tables:
                        rows=table.extract()
                        if len(rows)<2: continue
                        lines=[' | '.join((str(c).replace('\n',' ') if c else '') for c in row) for row in rows]
                        tabletext='\n'.join(lines)
                        if len(tabletext)>60:
                            tables.append(tabletext)
                except Exception as error:
                    print('table warning',source['code'],number,type(error).__name__,flush=True)
            for kind,content in [('text',text)]+[('table',t) for t in tables]:
                # Tables stay whole to preserve headers and column relationships.
                parts=[content] if kind=='table' else [content[i:i+1400] for i in range(0,len(content),1200)]
                for part in parts:
                    if len(part.strip())<40: continue
                    count+=1
                    chunks.append(dict(base,id=f"{source['code']}-p{number}-{count}",kind=kind,text=part))
        print(source['code'],len(doc),count,flush=True)
    for name,items in [('pages',pages),('chunks',chunks)]:
        with (ROOT/f'data/{name}.jsonl').open('w',encoding='utf-8') as f:
            for item in items: f.write(json.dumps(item,ensure_ascii=False)+'\n')
    print('DONE',len(pages),len(chunks),flush=True)

if __name__=='__main__': main()
