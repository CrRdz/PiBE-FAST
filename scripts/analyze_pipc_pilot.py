"""Audit the supplied workbook without modifying labels or imputing missing samples.
Run with the bundled Python runtime (openpyxl is used for read-only extraction).
"""
from pathlib import Path
from datetime import datetime
from collections import Counter
import csv, hashlib, json, re, statistics, subprocess
import openpyxl
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_data/pipc_pilot'
SOURCE = OUT / 'pipc_pilot_data.xlsx'

def encode(v):
    return v.isoformat(timespec='milliseconds') if isinstance(v, datetime) else v

def percentile(v, p):
    v=sorted(v)
    if not v:return None
    x=(len(v)-1)*p; i=int(x)
    return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)

def stats(v):
    return dict(n=len(v),mean=statistics.mean(v) if v else None,p50=percentile(v,.5),p95=percentile(v,.95),maximum=max(v) if v else None)

def main():
    wb=openpyxl.load_workbook(SOURCE,data_only=True)
    tables={}
    for sheet,file in [('Run_Metadata','runs'),('Event_Log','events'),('Resource_Log','resources')]:
        rows=list(wb[sheet].values);headers=rows[0]
        data=[dict(zip(headers,r),source_row=i) for i,r in enumerate(rows[1:],2) if r[0] is not None]
        tables[file]=data
        with (OUT/f'{file}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=[*headers,'source_row']);writer.writeheader()
            writer.writerows({k:encode(v) for k,v in r.items()} for r in data)
    runs,events,resources=(tables[k] for k in ('runs','events','resources'))
    assert len({r['run_id'] for r in runs})==len(runs)
    assert len({r['event_id'] for r in events})==len(events)
    assert all(r['run_id'] in {x['run_id'] for x in runs} for r in events+resources)
    latency_fields={'trigger_ms':('event_start_ts','trigger_created_ts'),
                    'transport_ms':('trigger_created_ts','pc_received_ts'),
                    'trigger_to_prompt_ms':('trigger_created_ts','prompt_displayed_ts'),
                    'event_to_final_ms':('event_start_ts','final_state_ts')}
    derived=[]
    for e in events:
        if e['trigger_generated'] is not True:continue
        row={'event_id':e['event_id'],'run_id':e['run_id'],'network_disconnect_injected':e['network_disconnect_injected']}
        for label,(a,b) in latency_fields.items():
            row[label]=(e[b]-e[a]).total_seconds()*1000 if isinstance(e[a],datetime) and isinstance(e[b],datetime) else None
            assert row[label] is None or row[label]>=0,(e['event_id'],label)
        derived.append(row)
    with (OUT/'derived_latency.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(derived[0]));writer.writeheader();writer.writerows(derived)
    result={'source_filename':'pipc_pilot_data.xlsx','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'scope':'author-supplied pilot workbook; implementation correspondence and original traces unresolved',
            'counts':{'runs':len(runs),'events':len(events),'resources':len(resources),'participants':len({r['participant_id'] for r in runs})},
            'total_run_hours':sum((r['run_end_ts']-r['run_start_ts']).total_seconds()/3600 for r in runs),
            'groups':{},'run_coverage':[],'integrity':{}}
    groups={'Earlier scripts':['run_001','run_002','run_003'],'Later scripts':['run_004','run_005','run_006'],'Recovery script':['run_007'],'Eight-hour run':['run_008']}
    for label,ids in groups.items():
        es=[e for e in events if e['run_id'] in ids]
        binary=[e for e in es if str(e['ground_truth_event']).lower() in ('true','false')]
        count=Counter(('T' if str(e['ground_truth_event']).lower()=='true' else 'F')+('P' if e['trigger_generated'] else 'N') for e in binary)
        # TP=true+trigger; TN=true+no trigger is a miss here, renamed explicitly.
        triggers=[e for e in es if e['trigger_generated'] is True]
        ls=[r for r in derived if r['run_id'] in ids]
        result['groups'][label]={'runs':ids,'events':len(es),'positive_events':count['TP']+count['TN'],'detected_positive':count['TP'],
                                'negative_events':count['FP']+count['FN'],'triggered_negative':count['FP'],
                                'excluded_nonbinary_events':len(es)-len(binary),'triggers':len(triggers),
                                'guided_requested':sum(e['guided_check_requested'] is True for e in es),
                                'guided_completed':sum(e['guided_check_requested'] is True and e['guided_check_completed'] is True for e in es),
                                'latency':{field:stats([r[field] for r in ls if r[field] is not None]) for field in latency_fields},
                                'latency_without_injected_disconnect':{field:stats([r[field] for r in ls if r['network_disconnect_injected'] is not True and r[field] is not None]) for field in latency_fields}}
    for r in runs:
        rs=sorted([x for x in resources if x['run_id']==r['run_id']],key=lambda x:x['sample_ts'])
        duration=(r['run_end_ts']-r['run_start_ts']).total_seconds();cadence=r['resource_sample_interval_s']
        diffs=Counter((b['sample_ts']-a['sample_ts']).total_seconds() for a,b in zip(rs,rs[1:]))
        result['run_coverage'].append(dict(run_id=r['run_id'],hours=duration/3600,samples=len(rs),expected_samples_half_open=round(duration/cadence),
            first_ts=encode(rs[0]['sample_ts']),last_ts=encode(rs[-1]['sample_ts']),cadence_s=cadence,interval_counts=dict(diffs),
            resource_stats={k:stats([x[k] for x in rs if isinstance(x[k],(float,int)) and not isinstance(x[k],bool)]) for k in ('pi_cpu_pct','pi_ram_used_mb','pi_temperature_c','pi_power_w','passive_b_actual_fps')},
            pi_offline_samples=sum(x['pi_online'] is False for x in rs),data_loss_flagged_samples=sum(x['data_loss_observed'] is True for x in rs),
            restart_counter_max=max(x['process_restart_count'] for x in rs)))
    commits=sorted({r['pi_app_commit'] for r in runs}|{r['pc_app_commit'] for r in runs})
    result['integrity']['commit_resolution']={c:subprocess.run(['git','cat-file','-e',c+'^{commit}'],cwd=ROOT,capture_output=True).returncode==0 for c in commits}
    result['integrity']['hash_token_lengths']=sorted({len(v.split('|')[0].strip().removeprefix('sha256:')) for r in runs for v in (r['pi_model_hashes'],r['pc_model_hashes'])})
    result['integrity']['sha256_tokens_valid']=all(re.fullmatch(r'sha256:[0-9a-fA-F]{64}',r[k].split('|')[0].strip()) is not None for r in runs for k in ('pi_model_hashes','pc_model_hashes'))
    result['integrity']['urgency_values']=dict(Counter(e['urgency_state'] for e in events))
    result['integrity']['external_alert_states']=dict(Counter(e['external_alert_delivery_state'] for e in events))
    result['integrity']['events_outside_run_bounds']=[e['event_id'] for e in events for r in runs if e['run_id']==r['run_id'] and (e['event_start_ts']<r['run_start_ts'] or e['event_end_ts']>r['run_end_ts'])]
    result['integrity']['manual_issues']=[
        'Recorded source references name github.com/example/pipc-monitoring and snapshots that are not supplied; commits remain unresolved locally.',
        'Configured 16 fps / 10 s windows, MobileNetV3 fall model and Whisper-tiny differ from the manuscript implementation.',
        'Structured protocol_defined sway labels are retained; narrative non-fall labels are not silently substituted.',
        'Original device logs, observer/video references, power-measurement method and clock synchronization trace are not attached.',
        'Some timestamp and resource measurements changed between workbook versions; the corrected workbook is used as supplied, without reconstructing the measurement process.']
    result['analyzer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('counts','total_run_hours','groups','integrity')},indent=2))

if __name__=='__main__':main()
