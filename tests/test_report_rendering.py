"""Execute the actual report renderers against a minimal DOM to check version labels."""
from pathlib import Path
import shutil
import subprocess
import unittest

class ReportRenderingTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for report rendering checks')
    def test_current_and_historical_report_renderers(self):
        html=(Path(__file__).resolve().parents[1]/'app/templates/index.html').read_text()
        history=html[html.index('    function renderHistory('):html.index('    async function openPersistentHistory(')]
        single=html[html.index('    function renderSingleReport('):html.index('    function render(status)')]
        script=r'''
const assert = require('node:assert/strict');
const elements = new Map();
function element() {return {textContent:'',children:[],replaceChildren(){this.children=[];},appendChild(e){this.children.push(e);}};}
const document = {getElementById(id){if(!elements.has(id)) elements.set(id,element());return elements.get(id);},createElement:element};
const language='en';
const old={id:1,component:'S',attempt:1,revision:1,decision:'clear',item:{status:'negative'}};
const current={id:2,component:'S',attempt:1,revision:2,decision:'emergency',item:{status:'negative',reported_functional_problem:true}};
const latest={befast:{reports:[old,current]}};
const t={historyAttempt:'{code} {attempt}',reportAttempt:'{name} {attempt}',itemNames:{S:'Speech'},decisions:{clear:'Clear',emergency:'Urgent'}};
function viewReport() {};
function resultDetail(){return 'Measurement interpretation';}
'''+history+single+r'''
renderHistory(latest.befast,t);
assert.equal(elements.get('reportHistory').children.length,2);
assert.match(elements.get('reportHistory').children[0].textContent,/Revision 2/);
renderSingleReport(current,t);
assert.equal(elements.get('decision').textContent,'Urgent');
assert.match(elements.get('reportMeta').textContent,/Revision 2/);
assert.doesNotMatch(elements.get('reportMeta').textContent,/Historical/);
assert.match(elements.get('results').children[0].children[0].textContent,/active symptom report/);
renderSingleReport(old,t);
assert.match(elements.get('reportMeta').textContent,/Historical snapshot/);
'''
        result=subprocess.run([shutil.which('node'),'-e',script],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
