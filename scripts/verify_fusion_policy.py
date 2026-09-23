"""Exhaustive synthetic decision-boundary verification, not sensor evaluation.

The complete-first variant suppresses urgency until all requested evidence is
complete. It is an experimental mechanism removal, not a clinical comparator
or either historical workbook scheme. No random sampling or human data.
"""
import csv,hashlib,itertools,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.befast.urgency import screening_assessment
CODES=('B','E','F','A','S')
STATES=('negative','insufficient','new_positive','longstanding_positive','unknown_positive')
def run():
    rows=[]
    for index,states in enumerate(itertools.product(STATES,repeat=5)):
        items={};onsets={}
        for code,state in zip(CODES,states):
            status='positive' if state.endswith('positive') else state
            items[code]={'status':status,'reason':'synthetic_policy_input'}
            onsets[code]={'new_positive':True,'longstanding_positive':False}.get(state)
        # Expected truth table is defined on fixture states, without the reducer.
        expected_u='urgent' if any(s in ('new_positive','unknown_positive') for s in states) else 'warning' if 'longstanding_positive' in states else 'none'
        expected_c='incomplete' if 'insufficient' in states else 'complete'
        actual=screening_assessment(items,onsets,CODES)
        delayed_u=actual['urgency'] if actual['completeness']=='complete' else 'none'
        rows.append(dict(case=index,states='|'.join(states),expected_urgency=expected_u,expected_completeness=expected_c,actual_urgency=actual['urgency'],actual_completeness=actual['completeness'],complete_first_urgency=delayed_u,full_pass=actual['urgency']==expected_u and actual['completeness']==expected_c,complete_first_pass=delayed_u==expected_u and actual['completeness']==expected_c))
    assert all(r['full_pass'] for r in rows)
    return rows
def main():
    out=ROOT/'experiments/fusion_policy_validation';out.mkdir(exist_ok=True);rows=run()
    with (out/'fusion_policy_cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'scope':'Exhaustive synthetic policy states at the decision boundary. B/E states denote eligible symptom evidence, not automatic eye classification. Not real-world detection or alert-rate evidence.', 'states_per_component':list(STATES),'components':list(CODES),'cases':len(rows),'full_passed':sum(r['full_pass'] for r in rows),'complete_first_passed':sum(r['complete_first_pass'] for r in rows),'withheld_urgency_cases':sum(not r['complete_first_pass'] for r in rows),'seed':None,'comparator_definition':'Same urgency/completeness outputs as Full, but urgency forced to none while completeness is incomplete. No historical comparator is reconstructed.','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'app/befast/urgency.py']}}
    (out/'fusion-policy-summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
