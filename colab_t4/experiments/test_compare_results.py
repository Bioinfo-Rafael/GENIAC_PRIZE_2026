import json,tempfile,unittest
from pathlib import Path
from compare_results import compare_results

class CompareTests(unittest.TestCase):
    def fixture(self,root):
        fields=('code_version','helpers_sha256','base_commit','split_sha256','eval_size','selected_ids_sha256','training_inventory','versions','gpu','config')
        for name,correct in [('run02',[True,False]),('run03',[False,True])]:
            path=root/name;path.mkdir()
            signature={k:'same' for k in fields};signature['model_sha256']=name
            rows=[dict(id=str(i),dataset=d,correct=c,text='private fixture') for i,(d,c) in enumerate(zip(['prm800k','gsm8k'],correct))]
            (path/'summary.json').write_text(json.dumps({'complete':True,'signature':signature,'overall':{'n':2,'correct':1},'by_dataset':{}}))
            (path/'predictions_private.json').write_text(json.dumps({'signature':signature,'rows':rows}))
        return root/'run02',root/'run03'
    def test_paired_comparison_excludes_text(self):
        with tempfile.TemporaryDirectory() as d:
            result=compare_results(*self.fixture(Path(d)))
            self.assertEqual(result['paired']['overall']['run03_only_correct'],1)
            self.assertEqual(result['paired']['prm800k']['accuracy_difference'],-1)
            self.assertNotIn('private fixture',json.dumps(result))
    def test_condition_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=self.fixture(Path(d));p=b/'summary.json';s=json.loads(p.read_text());s['signature']['config']='changed';p.write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,'conditions differ'):compare_results(a,b)
    def test_incomplete_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=self.fixture(Path(d));p=b/'summary.json';s=json.loads(p.read_text());s['complete']=False;p.write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,'complete'):compare_results(a,b)
    def test_swapped_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=self.fixture(Path(d));p=b/'predictions_private.json';s=json.loads(p.read_text());s['rows'].reverse();p.write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,'IDs differ'):compare_results(a,b)
    def test_generated_notebook(self):
        here=Path(__file__).parent;nb=json.loads((here/'compare_run02_run03_t4.ipynb').read_text())
        worker=(here/'compare_runs_worker.py').read_text()
        self.assertNotIn('drive.mount(',worker)
        self.assertNotIn('userdata.get(',worker)
        self.assertNotIn('compe_screening_eval',worker)
        self.assertIn('expected_rows":6000',worker)
        for c in nb['cells']:
            if c['cell_type']=='code':
                s=''.join(c['source']);s=s.split('\n',1)[1] if s.startswith('%%writefile') else s
                compile(s,c['id'],'exec');self.assertFalse(c['outputs'])
        import nbformat
        nbformat.validate(nbformat.from_dict(nb))
