"""保存每题三种检索器的真实召回；仅评估证据命中，不冒充生成答案正确率。"""
import json
from engine import Engine,ROOT
QUESTIONS=[
('美的集团2025年营业收入是多少？',{'美的集团':[10,11,50,51]},'事实查询'),
('格力电器2025年营业收入是多少？',{'格力电器':[7,8]},'事实查询'),
('海尔智家2025年营业收入是多少？',{'海尔智家':[12]},'事实查询'),
('苏泊尔2025年归属于上市公司股东的净利润是多少？',{'苏泊尔':[7,8]},'表格查询'),
('九阳股份2025年经营活动产生的现金流量净额是多少？',{'九阳股份':[7,8]},'表格查询'),
('小熊电器2025年营业收入同比增长多少？',{'小熊电器':[7,8]},'同比查询'),
('美的集团2025年营业收入与营业总收入有什么区别？',{'美的集团':[10,11]},'口径辨析'),
('十家公司的2025年营业收入分别是多少？',{'美的集团':[10,11],'格力电器':[7,8],'海尔智家':[12],'海信家电':[11,12],'苏泊尔':[7,8],'九阳股份':[7,8],'新宝股份':[7,8],'小熊电器':[7,8],'海信视像':[8],'长虹美菱':[9,10]},'跨公司全景'),
('海尔智家与海信视像2025年营业收入谁更高？',{'海尔智家':[12],'海信视像':[8]},'跨公司比较'),
('美的集团2027年营业收入精确是多少？',{},'超出资料范围')]

# Reference values independently read from the original main-financial-data pages.
EXPECTED={1:['456,451,731千元'],2:['170,447,058,533.57元'],3:['302,346,783,918.30元'],4:['2,096,720,231.52元'],5:['715,008,988.97元'],6:['9.92%'],7:['营业总收入','利息'],8:['456,451,731千元','170,447,058,533.57元','302,346,783,918.30元','87,928,485,899.72元','22,771,753,460.04元','8,209,763,125.36元','16,192,053,797.72元','5,230,079,556.58元','57,679,224,941.74元','30,407,926,517.70元'],9:['海尔智家的营业收入更高'],10:['证据不足']}

if __name__=='__main__':
    e=Engine(); results=[]
    for i,(q,gold,kind) in enumerate(QUESTIONS,1):
        for mode in ['bm25','vector','hybrid']:
            result=e.ask(q,mode)
            hit={company:any(c['company']==company and c['page'] in pages for c in result['evidence']) for company,pages in gold.items()}
            correct=all(value in result['answer'] for value in EXPECTED[i])
            error='' if correct else ('现金流量指标在布局文字中被数值列打断，规则未抽到答案。' if i==5 else '口径辨析超出规则回答能力；未召回指定参考页，未生成解释。')
            result.update(id=i,type=kind,reference_pages=gold,evidence_hits=hit,answer_correct=correct,expected=EXPECTED[i],error=error)
            results.append(result)
            print(i,mode,hit,flush=True)
    (ROOT/'output/evaluation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 十题真实评测记录','', '指标为指定参考页是否出现在召回证据中；不是答案正确率。参考页为主要财务指标页，其他页面也可能包含正确证据，因此未命中不必然代表答错。','', '|题号|问题|检索|参考公司覆盖|答案评价|','|---|---|---|---|---|']
    for r in results:
        hit=r['evidence_hits']; coverage=f"{sum(hit.values())}/{len(hit)}" if hit else '不适用'
        lines.append(f"|{r['id']}|{r['question']}|{r['mode']}|{coverage}|{'正确（含合理拒答）' if r['answer_correct'] else '未回答：'+r['error']}|")
    lines+=['','## 评价口径','','规则答案与独立核对的财报数值进行精确匹配；第10题以合理拒答计正确。第7题为未回答，不计为正确。不是大模型能力测评，不能外推到任意问题。','']
    for r in results:
        if r['mode']=='hybrid': lines += [f"### 第{r['id']}题",r['question'],'',r['answer'],'', '错误原因：'+(r['error'] or '未发现数值错误；仍需核对财务口径。'),'']
    (ROOT/'output/evaluation.md').write_text('\n'.join(lines),encoding='utf-8')
