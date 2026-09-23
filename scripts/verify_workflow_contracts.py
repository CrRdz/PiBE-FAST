"""Bounded software contract checks using real sessions and synthetic results.

No sensors, human participants, hardware timing, or clinical labels are used.
Each row is a checked configuration, not an independent statistical trial.
"""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
from dataclasses import replace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.befast.session import BefastSession
from app.befast.result import MotionResult, motion_report_item
from app.befast.fusion import build_feature_fusion
from app.befast.report import eye_report_item
from app.befast.config import BefastConfig
from app.befast.urgency import screening_assessment


def negative(**kwargs):
    return MotionResult(status='negative', reason='synthetic_qualified_negative', quality=1, **kwargs)


def run_cases():
    rows = []
    def checked(group, scenario, expected, actual):
        assert expected == actual, (group, scenario, expected, actual)
        rows.append(dict(group=group, scenario=scenario, expected=str(expected), actual=str(actual), passed=True))

    for answer, camera_ok in itertools.product((None, False, True), (False, True)):
        s = BefastSession(); s.start_screening()
        s.manual['balance_problem'] = False; s.manual_completed['B'] = True
        for attr in ('balance_result', 'face_result', 'arm_result', 'speech_result'):
            setattr(s, attr, negative())
        if answer is not None:
            s.submit_component_observation('E', answer, True)
        s.eye_result = MotionResult(status='insufficient', reason='eye_metrics_recorded_for_validation' if camera_ok else 'camera_failed', quality=.9 if camera_ok else 0, metrics={'left_gaze_range':.4} if camera_ok else {})
        result = s.snapshot()
        expected = ('urgent' if answer is True else 'none', 'incomplete' if answer is None else 'complete')
        checked('E completion', f'answer={answer}, camera_ok={camera_ok}', expected, (result['urgency'], result['completeness']))

    # Independent explicit-null baseline: all requested fields must be present.
    for statuses in itertools.product(('negative', 'positive', None), repeat=3):
        items = {c: {'status': v, 'reason': 'synthetic'} for c,v in zip(('F','A','S'), statuses) if v is not None}
        expected = ('urgent' if 'positive' in statuses else 'none', 'incomplete' if None in statuses else 'complete')
        result = screening_assessment(items, None, ('F','A','S'))
        checked('Nullable baseline', str(statuses), expected, (result['urgency'], result['completeness']))

    for code, onset, order in itertools.product(('F','A','S'), (True, False, None), ('before_retry','after_retry')):
        s = BefastSession(); s.start_screening()
        if order == 'before_retry': s.report_functional_problem(code, onset)
        s.prepare_component(code)
        if order == 'after_retry': s.report_functional_problem(code, onset)
        s.set_component_onset(code, False)
        result = s.snapshot()
        checked('Report persistence', f'{code}, onset={onset}, {order}', ('warning' if onset is False else 'urgent', 'incomplete'), (result['urgency'], result['completeness']))
        event = result['items'][code]['symptom_evidence'][0]
        s.prepare_component(code)
        checked('Evidence identity', f'{code}, onset={onset}, {order}', event, s.snapshot()['items'][code]['symptom_evidence'][0])
        s.reset(); s.start_screening()
        checked('Reset isolation', f'{code}, onset={onset}, {order}', [], s.snapshot()['positive_components'])

    for report_onset, audio_onset, audio_positive in itertools.product((True,False,None), (True,False,None), (False,True)):
        for report_first in (True, False):
            s = BefastSession(); s.prepare_component('S'); token=s.start_speech_recording()
            r = MotionResult(status='positive' if audio_positive else 'negative', reason='synthetic_audio', quality=1, details={'speech_attempt_id':token})
            if report_first: s.report_functional_problem('S',report_onset)
            s.submit_speech_result(r,new_or_sudden=audio_onset)
            if not report_first: s.report_functional_problem('S',report_onset)
            expected='urgent' if report_onset is not False or (audio_positive and audio_onset is not False) else 'warning'
            checked('Speech ordering', f'report={report_onset}, audio={audio_onset}, positive={audio_positive}, report_first={report_first}', expected, s.snapshot()['urgency'])

    for transition in ('retry','reset','duplicate'):
        s=BefastSession(); s.prepare_component('S'); old=s.start_speech_recording()
        r=negative(details={'speech_attempt_id':old})
        if transition=='duplicate': s.submit_speech_result(r,new_or_sudden=False)
        else:
            if transition=='reset': s.reset()
            s.prepare_component('S'); s.start_speech_recording()
        before=s.snapshot()
        rejected=False
        try: s.submit_speech_result(r,new_or_sudden=False)
        except ValueError: rejected=True
        checked('Stale result rejection',transition,True,rejected)
        after=s.snapshot()
        checked('Stale result isolation',transition,(before['stage'],before['items'],before['urgency']), (after['stage'],after['items'],after['urgency']))

    camera=MotionResult(status='insufficient',reason='eye_metrics_recorded_for_validation',quality=.9,metrics={'left_gaze_range':.4,'right_gaze_range':.4,'minimum_trial_valid_fraction':.9,'minimum_response_snr':10,'max_repeat_relative_error':0})
    config=BefastConfig()
    f=motion_report_item(negative(metrics={'mouth_corner_delta':.075,'smile_score_difference':0,'smile_strength':.4}),'synthetic')
    reference=build_feature_fusion({'E':eye_report_item(False,True,camera),'F':f},config)
    for answer in (None,False,True):
        e=eye_report_item(answer,answer is not None,camera)
        fusion=build_feature_fusion({'E':e,'F':f},config)
        checked('Measurement preservation',f'E answer={answer}',reference['model_vector'],fusion['model_vector'])
        checked('Undefined severity',f'E answer={answer}',(None,1,1.0),(fusion['domain_severity']['E'],fusion['aggregate']['reliable_domain_count'],fusion['aggregate']['mean_top_two_domain_severity']))
    missing=build_feature_fusion({'S':motion_report_item(negative(metrics={'pause_fraction':.2}),'synthetic')},config)
    checked('Undefined severity','S probability absent',(None,None),(missing['domain_severity']['S'],missing['aggregate']['mean_top_two_domain_severity']))
    return rows


def main():
    from importlib.util import spec_from_file_location, module_from_spec
    spec=spec_from_file_location("lifecycle_checks", ROOT/"tests/test_evidence_lifecycle.py")
    module=module_from_spec(spec);spec.loader.exec_module(module)
    rows=run_cases()+module.lifecycle_cases(); out=ROOT/'experiments/software_validation'; out.mkdir(exist_ok=True)
    with (out/'state_cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    groups={g:sum(r['group']==g for r in rows) for g in dict.fromkeys(r['group'] for r in rows)}
    paths=[Path(__file__),*[ROOT/p for p in ('app/befast/session.py','app/befast/report.py','app/befast/urgency.py','app/befast/fusion.py','app/speech_audio.py','app/web.py','app/history.py','app/befast/balance.py','tests/test_evidence_lifecycle.py')]]
    summary={'scope':'bounded synthetic software configurations; no hardware or clinical experiment', 'checks':len(rows),'passed':sum(r['passed'] for r in rows),'groups':groups,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (out/'state_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
