import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
import run02_launcher as r


class LauncherTests(unittest.TestCase):
    def make_zip(self, path, files):
        with ZipFile(path,'w') as z:
            z.writestr('bundle_manifest.json',json.dumps({'files':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()}}))
            for k,v in files.items():z.writestr(k,v)
        return r.sha256_file(path)

    def test_restore_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); z=root/'input.zip'
            sha=self.make_zip(z,{'data_cache/fixture':b'fixture'})
            self.assertEqual(r.restore_bundle(z,root/'drive',sha)['added'],1)
            self.assertEqual(r.restore_bundle(z,root/'drive',sha)['already_present'],1)

    def test_conflict_does_not_write_any_pending_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); z=root/'input.zip'; dest=root/'drive';dest.mkdir()
            (dest/'existing').write_bytes(b'user content')
            sha=self.make_zip(z,{'new':b'fixture','existing':b'different'})
            with self.assertRaises(RuntimeError):r.restore_bundle(z,dest,sha)
            self.assertFalse((dest/'new').exists())
            self.assertEqual((dest/'existing').read_bytes(),b'user content')

    def test_wrong_archive_and_path_traversal_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); z=root/'input.zip'
            sha=self.make_zip(z,{'../outside':b'fixture'})
            with self.assertRaises(RuntimeError):r.restore_bundle(z,root/'drive','bad hash')
            with self.assertRaises(RuntimeError):r.restore_bundle(z,root/'drive',sha)
            self.assertFalse((root/'outside').exists())

    def test_child_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'log'
            r.run_logged([sys.executable,'-c','print("fixture")'],path,echo=False)
            self.assertIn('fixture',path.read_text())
            with self.assertRaises(RuntimeError):
                r.run_logged([sys.executable,'-c','raise SystemExit(4)'],path,echo=False)

    def test_complete_requires_matching_saved_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run='fixture'; report=root/'experiments'/run/'training_summary.json'
            self.assertFalse(r.completed_training(root,run))
            adapter=root/'adapters'/(run+'_stage2')/'adapter_model.safetensors'
            adapter.parent.mkdir(parents=True);adapter.write_bytes(b'opaque fixture, not model weights')
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({'complete':True,'stage2_weights_sha256':r.sha256_file(adapter)}))
            self.assertTrue(r.completed_training(root,run))
            adapter.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):r.completed_training(root,run)

    def test_generated_notebook_and_worker(self):
        root=Path(__file__).parent
        nb=json.loads((root/'run02_auto_t4.ipynb').read_text())
        for c in nb['cells']:
            if c['cell_type']=='code':
                s=''.join(c['source'])
                if s.startswith('%%writefile'):s=s.split('\n',1)[1]
                compile(s,c['id'],'exec')
                self.assertFalse(c['outputs'])
        worker=(root/'run02_worker.py').read_text()
        self.assertNotIn('drive.mount(',worker)
        self.assertIn('PHASE = sys.argv[1]',worker)
        self.assertNotIn('compe_screening_eval',worker)
        import importlib.util
        if importlib.util.find_spec('nbformat'):
            import nbformat
            nbformat.validate(nbformat.from_dict(nb))

if __name__=='__main__':unittest.main()
