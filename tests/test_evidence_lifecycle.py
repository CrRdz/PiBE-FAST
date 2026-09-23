"""Cross-modality lifecycle contracts with synthetic measurements and real sessions."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import tempfile
import unittest
from app.befast.session import BefastSession
from app.befast.result import MotionResult
from app.befast.fusion import build_feature_fusion
from app.history import AbnormalHistoryStore

CODES = ('B','E','F','A','S')
ATTR = dict(zip(CODES, ('balance_result','eye_result','face_result','arm_result','speech_result')))
METRICS = {
    'B': {'trunk_orientation_change_score':1.,'mediolateral_sway_change_score':1.},
    'E': {'left_gaze_range':.4,'minimum_trial_valid_fraction':.9,'minimum_response_snr':10,'max_repeat_relative_error':0},
    'F': {'mouth_corner_delta':.01,'smile_score_difference':.01,'smile_strength':.4},
    'A': {'level_difference':.01,'drift_difference':.01},
    'S': {'mdsc_dysarthria_probability':.1},
}

def report(s, code, onset):
    if code in ('B','E'): s.submit_component_observation(code,True,onset)
    else: s.report_functional_problem(code,onset)

def initialized(code, status='negative'):
    s=BefastSession(); s.start_screening(); s.active_component=code
    reason='eye_metrics_recorded_for_validation' if code=='E' and status=='negative' else 'synthetic'
    setattr(s,ATTR[code],MotionResult(status=status,reason=reason,quality=1,metrics=METRICS[code]))
    return s

def lifecycle_cases():
    rows=[]
    def check(group, scenario, expected, actual):
        assert expected==actual,(group,scenario,expected,actual)
        rows.append(dict(group=group,scenario=scenario,expected=str(expected),actual=str(actual),passed=True))
    for code in CODES:
        for onset in (True,False,None):
            for status in ('negative','insufficient','not_run'):
                s=initialized(code,status); report(s,code,onset)
                event=s.snapshot()['items'][code]['symptom_evidence'][0]
                s.prepare_component(code)
                check('All-modal retry retention',f'{code}/{onset}/{status}','warning' if onset is False else 'urgent',s.snapshot()['urgency'])
                s.retract_symptom(event['event_id'],'Synthetic entry correction')
                p=s.snapshot()
                check('All-modal corrections',f'{code}/{onset}/{status}',('none',False),(p['urgency'],p['items'][code]['symptom_evidence'][0]['active']))
            s=initialized(code); before=s.snapshot()['fusion']['model_vector']
            report(s,code,onset)
            after=build_feature_fusion(s.snapshot()['items'],s.config)['model_vector']
            check('All-modal measurements',f'{code}/{onset}',before,after)
        s=initialized(code); report(s,code,True); old=deepcopy(s.snapshot()['reports'][-1])
        event=s.snapshot()['items'][code]['symptom_evidence'][0]
        s.retract_symptom(event['event_id'],'Correction test')
        p=s.snapshot(); new=p['reports'][-1]
        check('Versioned corrections',code,(old['id'],old['revision']+1,'none'),(new['supersedes'],new['revision'],new['urgency']))
        # Mutating API snapshots must not mutate live or previously recorded events.
        p['reports'][0]['item']['symptom_evidence'][0]['active']=False
        check('Historical snapshot isolation',code,old,s.snapshot()['reports'][0])
        with tempfile.TemporaryDirectory() as directory:
            store=AbnormalHistoryStore(directory)
            a=store.save_positive_report(old,None,1)
            b=store.save_positive_report(new,None,2)
            check('Persistent correction audit',code,(old,new),(a['details']['evidence_report'],b['details']['evidence_report']))
        s=BefastSession();s.start_screening()
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda onset: report(s,code,onset),(True,False,None)))
        p=s.snapshot()
        check('Concurrent symptom updates',code,(3,'urgent',[1,2,3]),(len(p['items'][code]['symptom_evidence']),p['urgency'],[r['revision'] for r in p['reports']]))
    for status,expected in (('not_run','complete'),('insufficient','incomplete')):
        s=initialized('B',status);report(s,'B',True)
        check('Standing safety exemption',status,expected,s.snapshot()['reports'][-1]['completeness'])
    return rows

class EvidenceLifecycleTest(unittest.TestCase):
    def test_balance_retry_exempts_new_attempt_and_retains_failed_report(self):
        s = initialized('B', 'insufficient')
        report(s, 'B', True)
        old = deepcopy(s.snapshot()['reports'][-1])
        self.assertEqual(old['completeness'], 'incomplete')
        s.prepare_component('B')
        s.submit_component_observation('B', False, False)
        p = s.snapshot()
        self.assertEqual(p['stage'], 'report')
        self.assertEqual(p['items']['B']['completion_status'], 'safety_exempt')
        self.assertEqual(p['reports'][-1]['completeness'], 'complete')
        self.assertEqual(p['urgency'], 'urgent')
        self.assertEqual(p['reports'][0], old)
        self.assertEqual(len(p['items']['B']['symptom_evidence']), 1)

    def test_active_balance_symptom_blocks_direct_standing_start(self):
        s = initialized('B', 'not_run')
        report(s, 'B', True)
        s.prepare_component('B')
        with self.assertRaises(ValueError):
            s.start_stage('balance')
        self.assertEqual(s.balance_result.status, 'not_run')

    def test_correcting_balance_symptom_allows_fresh_negative_answer_to_prepare(self):
        s = initialized('B', 'not_run')
        report(s, 'B', True)
        event = s.snapshot()['items']['B']['symptom_evidence'][0]
        s.prepare_component('B')
        s.retract_symptom(event['event_id'], 'Mistaken entry')
        s.submit_component_observation('B', False, False)
        self.assertEqual(s.snapshot()['stage'], 'ready_balance')
        s.start_stage('balance')
        self.assertEqual(s.balance_result.status, 'checking')

    def test_correction_preserves_independent_measured_positive(self):
        for code in ('B', 'F', 'A', 'S'):
            with self.subTest(component=code):
                s = initialized(code, 'positive')
                s.set_component_onset(code, True)
                report(s, code, False)
                event = s.snapshot()['items'][code]['symptom_evidence'][0]
                s.retract_symptom(event['event_id'], 'Incorrect symptom entry')
                self.assertEqual(s.snapshot()['urgency'], 'urgent')
                self.assertEqual(s.snapshot()['items'][code]['status'], 'positive')

    def test_correction_of_one_event_retains_another_active_event(self):
        for code in CODES:
            with self.subTest(component=code):
                s = initialized(code)
                report(s, code, False)
                first = s.snapshot()['items'][code]['symptom_evidence'][0]
                report(s, code, True)
                s.retract_symptom(first['event_id'], 'First entry corrected')
                self.assertEqual(s.snapshot()['urgency'], 'urgent')
                self.assertEqual(sum(e['active'] for e in s.snapshot()['items'][code]['symptom_evidence']), 1)

    def test_retry_and_symptom_submission_interleavings_preserve_evidence(self):
        for code in CODES:
            for symptom_first in (True, False):
                with self.subTest(component=code, symptom_first=symptom_first):
                    s = initialized(code)
                    actions = [lambda: report(s, code, True), lambda: s.prepare_component(code)]
                    if not symptom_first:
                        actions.reverse()
                    for action in actions:
                        action()
                    self.assertEqual(s.snapshot()['urgency'], 'urgent')
                    self.assertEqual(len(s.snapshot()['items'][code]['symptom_evidence']), 1)

    def test_cross_modal_matrix(self):
        self.assertEqual(len(lifecycle_cases()),127)

    def test_late_symptom_revises_current_report_without_changing_old_report(self):
        s=initialized('S');s.prepare_component('S');token=s.start_speech_recording()
        s.submit_speech_result(MotionResult(status='negative',reason='synthetic',details={'speech_attempt_id':token}),new_or_sudden=False)
        old=s.snapshot()['current_report'];s.report_functional_problem('S',True);p=s.snapshot()
        self.assertEqual(p['current_report']['decision'],'emergency')
        self.assertEqual(p['current_report']['supersedes'],old['id'])
        self.assertEqual(p['reports'][0],old)
        self.assertEqual(p['current_report']['new_or_sudden'],True)

    def test_measurement_onset_updates_revise_current_report(self):
        s=initialized('S');s.prepare_component('S');token=s.start_speech_recording()
        s.submit_speech_result(MotionResult(status='positive',reason='synthetic',details={'speech_attempt_id':token}),new_or_sudden=True)
        old=s.snapshot()['current_report']
        s.set_component_onset('S',False)
        current=s.snapshot()['current_report']
        self.assertEqual(current['decision'],'warning')
        self.assertEqual(current['supersedes'],old['id'])
        self.assertEqual(s.snapshot()['reports'][0]['decision'],'emergency')

    def test_balance_answer_does_not_overwrite_acquired_measurement_onset(self):
        for answer in (True, False):
            s=initialized('B','positive')
            s.set_component_onset('B',True)
            s.submit_component_observation('B',answer,False)
            self.assertIs(s.snapshot()['component_onsets']['B'],True)
            self.assertEqual(s.snapshot()['urgency'],'urgent')

    def test_correction_requires_active_event_and_nonempty_reason(self):
        s=initialized('F');report(s,'F',True);event=s.snapshot()['items']['F']['symptom_evidence'][0]
        with self.assertRaises(ValueError):s.retract_symptom(event['event_id'],' ')
        self.assertEqual(s.snapshot()['urgency'],'urgent')
        s.retract_symptom(event['event_id'],'mistaken entry')
        with self.assertRaises(ValueError):s.retract_symptom(event['event_id'],'again')

if __name__=='__main__':unittest.main()
