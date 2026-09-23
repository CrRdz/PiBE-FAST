"""Audit exported historical evidence without inferring missing runtime paths."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/validation_study'
DEST=ROOT/'experiments/fusion_policy_validation'
def read(name):
    return list(csv.DictReader((SOURCE/name).open(encoding='utf-8-sig')))
def main():
    DEST.mkdir(exist_ok=True)
    sessions={r['会话ID']:r for r in read('sessions.csv')}
    outputs={r['事件ID']:r for r in read('system_outputs.csv') if r['方案ID']=='SCH-FULL'}
    rows=[]
    for e in read('events_and_references.csv'):
        if e['参考目标事件']=='无':continue
        o=outputs[e['事件ID']]
        rows.append(dict(event_id=e['事件ID'],component=e['组件'],session_category=sessions[e['会话ID']]['采集队列'],recorded_operation=e['操作类型'],recorded_outcome=o['测量状态'],runtime_decision_path='unresolved',runtime_path_evidence='',raw_log_index=e['原始事件日志索引'],interpretation='recorded target response only'))
    with (DEST/'target_provenance.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'scope':'No inference from component, timestamp, or operation label to runtime path.', 'targets':len(rows),'triggered':sum(r['recorded_outcome']=='已触发' for r in rows),'runtime_paths_resolved':0,'historical_comparators':{'SCH-A-NOBASE':{'population_parameters':None,'construction_provenance':None},'SCH-B-NOGATE':{'disabled_rules':None}},'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.csv')}}
    (DEST/'historical-evidence-audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
