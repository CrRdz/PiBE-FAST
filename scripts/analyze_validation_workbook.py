"""Read-only, row-level audit of the 2026 validation workbook.

Run with bundled Python. No source values are corrected or imputed.
Bootstrap samples participants with replacement and retains paired schemes.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import shutil
import argparse
from datetime import datetime

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=ROOT / 'experiments/validation_study/pibefast_validation_data.xlsx')
parser.add_argument('--output', type=Path, default=ROOT / 'experiments/validation_study')
args = parser.parse_args()
SOURCE = args.source.resolve()
DEST = args.output.resolve()
DEST.mkdir(parents=True, exist_ok=True)
ARCHIVE = DEST / 'pibefast_validation_data.xlsx'
if SOURCE != ARCHIVE:
    shutil.copy2(SOURCE, ARCHIVE)
book = openpyxl.load_workbook(SOURCE, data_only=True)
tables = {}
EXPORT_FILENAMES = {
    '会话登记': 'sessions.csv',
    '事件与参考': 'events_and_references.csv',
    '系统输出': 'system_outputs.csv',
    'FE参考测量': 'face_eye_reference_measurements.csv',
}
for name in ('会话登记', '事件与参考', '系统输出', 'FE参考测量'):
    values = list(book[name].values)
    tables[name] = [dict(zip(values[0], row), source_row=i)
                    for i, row in enumerate(values[1:], 2) if row[0]]
    with (DEST / EXPORT_FILENAMES[name]).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[*values[0], 'source_row'])
        writer.writeheader()
        writer.writerows(tables[name])
sessions = {r['会话ID']: r for r in tables['会话登记']}
events = {r['事件ID']: r for r in tables['事件与参考']}
schemes = ['SCH-FULL', 'SCH-A-NOBASE', 'SCH-B-NOGATE']
outputs = collections.defaultdict(dict)
for r in tables['系统输出']:
    assert r['方案ID'] not in outputs[r['事件ID']]
    outputs[r['事件ID']][r['方案ID']] = r
assert set(outputs) == set(events)
assert all(set(v) == set(schemes) for v in outputs.values())
assert all(e['会话ID'] in sessions for e in events.values())

def participant(e):
    return sessions[e['会话ID']]['参与者ID']

def paired(rows, outcome):
    ids = sorted({participant(e) for e in rows})
    counts = np.zeros((len(ids), 4))
    for e in rows:
        i = ids.index(participant(e))
        counts[i, 3] += 1
        for k, scheme in enumerate(schemes):
            counts[i, k] += outcome(e, outputs[e['事件ID']][scheme])
    estimates = counts[:, :3].sum(0) / counts[:, 3].sum()
    rng = np.random.default_rng(20260910)
    sampled = counts[rng.integers(len(ids), size=(10000, len(ids)))].sum(1)
    rates = sampled[:, :3] / sampled[:, 3:4]
    return dict(n=len(rows), participants=len(ids), ids=[e['事件ID'] for e in rows],
                numerator=counts[:, :3].sum(0).astype(int).tolist(),
                rate=estimates.tolist(),
                rate_ci=np.quantile(rates, [.025, .975], axis=0).T.tolist(),
                difference=(estimates[0] - estimates[1:]).tolist(),
                difference_ci=np.quantile(rates[:, :1] - rates[:, 1:], [.025, .975], axis=0).T.tolist())

target = [e for e in events.values() if e['参考目标事件'] != '无']
incomplete = [e for e in events.values() if e['参考完整性'] == '不完整']
nonabort = [e for e in incomplete if outputs[e['事件ID']][schemes[0]]['测量状态'] != '中止']
correction = [e for e in events.values() if '纠正' in e['症状报告内容']]
started = [e for e in events.values() if e['操作类型'] != '连续观察触发']
natural = [s for s in sessions.values() if s['采集队列'] == '自然使用']
natural_ids = {s['会话ID'] for s in natural}
false_events = [e for e in events.values() if e['会话ID'] in natural_ids and e['操作类型'] == '连续观察触发']
urgency_rank = {'无': 0, 'none': 0, '关注': 1, 'warning': 1, 'urgent': 2}
def urgency_shortfall(e, o):
    return urgency_rank[o['系统紧急程度']] < urgency_rank[e['参考紧急程度']]
result = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    schemes=schemes, bootstrap=dict(resamples=10000, seed=20260910, unit='participant', method='percentile'),
    inventory=dict(participants=len({s['参与者ID'] for s in sessions.values()}), sessions=len(sessions), events=len(events), outputs=sum(map(len,outputs.values())), reference_rows=len(tables['FE参考测量']),
                   session_cohorts=dict(collections.Counter(s['采集队列'] for s in sessions.values())),
                   dates=[min(s['开始时间_UTC'] for s in sessions.values()), max(s['结束时间_UTC'] for s in sessions.values())]),
    detection=paired(target, lambda e,o: o['测量状态']=='已触发'),
    urgency_shortfall=paired(target, urgency_shortfall),
    incorrect_complete_all=paired(incomplete, lambda e,o: o['系统完整性']=='完整'),
    incorrect_complete_excluding_abort=paired(nonabort, lambda e,o: o['系统完整性']=='完整'),
    correction_history=paired(correction, lambda e,o: o['纠正历史完整']=='是'))
result['component_detection'] = {c: paired([e for e in target if e['组件']==c], lambda e,o: o['测量状态']=='已触发') for c in sorted({e['组件'] for e in target})}
result['urgency_definition'] = 'Output urgency below event-specific reference level among all target events, including undetected targets.'
result['detected_urgency_audit'] = {}
for sc in schemes:
    detected = [e for e in target if outputs[e['事件ID']][sc]['测量状态']=='已触发']
    result['detected_urgency_audit'][sc] = dict(n=len(detected),
        mismatch_ids=[e['事件ID'] for e in detected if urgency_rank[outputs[e['事件ID']][sc]['系统紧急程度']] != urgency_rank[e['参考紧急程度']]])
result['retained_symptoms'] = {}
for sc in schemes:
    known=[e for e in target if outputs[e['事件ID']][sc]['保留症状数'] is not None]
    fewer=[e['事件ID'] for e in known if outputs[e['事件ID']][sc]['保留症状数'] < e['应保留症状数']]
    zero=[e['事件ID'] for e in known if outputs[e['事件ID']][sc]['保留症状数']==0]
    result['retained_symptoms'][sc]=dict(known=len(known), missing=[e['事件ID'] for e in target if e not in known], fewer_than_reference=fewer, zero=zero)
result['acquisition']=dict(attempts=len(started), reference_complete=sum(e['参考完整性']=='完整' for e in started),
    participants=len({participant(e) for e in started}), states={sc:dict(collections.Counter(outputs[e['事件ID']][sc]['测量状态'] for e in started)) for sc in schemes})
exposure = {k:sum(s[k] for s in natural)/3600 for k in ('计划观察时长_s','系统在线时长_s','参考阴性且在线_s')}
fp_counts = [sum(outputs[e['事件ID']][sc]['测量状态']=='已触发' for e in false_events) for sc in schemes]
result['natural_use']=dict(sessions=len(natural),participants=len({s['参与者ID'] for s in natural}),hours=exposure,
    triggers=fp_counts, per_hour=(np.array(fp_counts)/exposure['参考阴性且在线_s']).tolist(),
    availability=exposure['系统在线时长_s']/exposure['计划观察时长_s'])
natural_counts=np.array([[sum(outputs[e['事件ID']][sc]['测量状态']=='已触发' for e in false_events if e['会话ID']==s['会话ID']) for sc in schemes]+[s['参考阴性且在线_s']/3600] for s in natural])
rng=np.random.default_rng(20260910)
boot=natural_counts[rng.integers(len(natural),size=(10000,len(natural)))].sum(1)
rates=boot[:,:3]/boot[:,3:4]
result['natural_use']['rate_ci']=np.quantile(rates,[.025,.975],axis=0).T.tolist()
result['natural_use']['difference_ci']=np.quantile(rates[:,:1]-rates[:,1:],[.025,.975],axis=0).T.tolist()
result['timing'] = {}
for c in sorted({e['组件'] for e in target}):
    es=[e for e in target if e['组件']==c and outputs[e['事件ID']][schemes[0]]['测量状态']=='已触发']
    delays=[outputs[e['事件ID']][schemes[0]]['决策_相对秒']-e['事件起始_相对秒'] for e in es]
    premature=[e['事件ID'] for e in es if outputs[e['事件ID']][schemes[0]]['决策_相对秒']<outputs[e['事件ID']][schemes[0]]['采集结束_相对秒']]
    result['timing'][c]=dict(n=len(es),median=float(np.median(delays)),p95=float(np.quantile(delays,.95)),decision_before_capture_end=premature)
refs = tables['FE参考测量']
result['reference_agreement']={}
for metric in sorted({r['指标名称'] for r in refs}):
    rs=[r for r in refs if r['指标名称']==metric and r['不可比较原因'] is None and isinstance(r['独立参考值'],(int,float))]
    if not rs: continue
    errors=np.array([r['系统值']-r['独立参考值'] for r in rs])
    result['reference_agreement'][metric]=dict(n=len(rs),participants=len({participant(events[r['事件ID']]) for r in rs}),mae=float(np.abs(errors).mean()),bias=float(errors.mean()),min=float(min(r['系统值'] for r in rs)),max=float(max(r['系统值'] for r in rs)),units=list({r['单位'] for r in rs}))
# Chronological pairs: only matching participant, metric and explicit repeat indices 1/2.
repeat=collections.defaultdict(dict)
duplicates=[]
for r in refs:
    if r['指标名称']=='嘴角变化差' and r['重复次序'] in (1,2):
        p=participant(events[r['事件ID']])
        if r['重复次序'] in repeat[p]: duplicates.append(r['测量ID'])
        repeat[p][r['重复次序']]=r
pairs={p:v for p,v in repeat.items() if set(v)=={1,2}}
x=np.array([[v[k]['系统值'] for k in (1,2)] for v in pairs.values()])
n,k=x.shape
msr=k*np.var(x.mean(1),ddof=1); msc=n*np.var(x.mean(0),ddof=1)
mse=np.square(x-x.mean(1,keepdims=True)-x.mean(0,keepdims=True)+x.mean()).sum()/((n-1)*(k-1))
icc=(msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)
result['face_repeatability']=dict(pairs=n,icc21=float(icc),duplicate_rows=duplicates,ids={p:[v[t]['事件ID'] for t in (1,2)] for p,v in pairs.items()})
boot=x[np.random.default_rng(20260910).integers(n,size=(10000,n))]
msr=k*np.var(boot.mean(2),axis=1,ddof=1)
msc=n*np.var(boot.mean(1),axis=1,ddof=1)
mse=np.square(boot-boot.mean(2,keepdims=True)-boot.mean(1,keepdims=True)+boot.mean((1,2),keepdims=True)).sum((1,2))/((n-1)*(k-1))
iccs=(msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)
result['face_repeatability']['ci']=np.quantile(iccs,[.025,.975]).tolist()
intervals=[]
for v in pairs.values():
    times=[]
    for t in (1,2):
        e=events[v[t]['事件ID']]
        times.append(datetime.fromisoformat(sessions[e['会话ID']]['开始时间_UTC']).timestamp()+e['事件起始_相对秒'])
    intervals.append((times[1]-times[0])/86400)
result['face_repeatability']['interval_days_range']=[min(intervals),max(intervals)]
result['audit']={
    'reference_rows_on_reference_incomplete_events':[r['测量ID'] for r in refs if events[r['事件ID']]['参考完整性']=='不完整'],
    'duplicate_sequence_steps':{sid:[v for v,c in collections.Counter(e['序列步骤'] for e in events.values() if e['序列ID']==sid).items() if c>1] for sid in sorted({e['序列ID'] for e in events.values()})},
    'session_duration_mismatch':[s['会话ID'] for s in sessions.values() if (datetime.fromisoformat(s['结束时间_UTC'])-datetime.fromisoformat(s['开始时间_UTC'])).total_seconds()!=s['计划观察时长_s']],
    'events_beyond_session':[e['事件ID'] for e in events.values() if e['事件结束_相对秒']>sessions[e['会话ID']]['计划观察时长_s']],
    'face_counts_above_5hz_protocol':[r['测量ID'] for r in refs if r['组件']=='Face' and r['有效样本数']>30],
    'arm_protocols':sorted({r['动作或目标阶段'] for r in refs if r['组件']=='Arms'}),
}
(DEST/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('audit','retained_symptoms')},ensure_ascii=False,indent=2))
