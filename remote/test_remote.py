"""Exercise orchestration without downloading a model or touching a GPU."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run02
import build_worker


class RemoteRunTests(unittest.TestCase):
    def test_worker_is_current_and_compiles(self):
        before = (build_worker.HERE / 'run02_worker.py').read_text()
        build_worker.build()
        after = (build_worker.HERE / 'run02_worker.py').read_text()
        self.assertEqual(before, after)
        compile(after, 'remote-worker', 'exec')
        self.assertNotIn('google.colab', after)
        self.assertNotIn('/content/', after)

    def test_mac_does_not_start_training(self):
        with patch('sys.argv', ['run02']), patch.object(run02.platform, 'system', return_value='Darwin'), \
                patch.object(run02, 'run_logged') as child, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run02.main()
            child.assert_not_called()

    def test_completed_run_does_not_install_or_launch(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sys.argv', ['run02', '--workspace', tmp, '--install']), \
                patch.object(run02.platform, 'system', return_value='Linux'), \
                patch.object(run02, 'completed_training', return_value=True), \
                patch.object(run02, 'run_logged') as child, contextlib.redirect_stdout(io.StringIO()):
            run02.main()
            child.assert_not_called()

    def test_merge_failure_prevents_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / 'adapters/t4_20261013_run01_stage1/adapter_model.safetensors'
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(b'test fixture')
            with patch('sys.argv', ['run02', '--workspace', tmp]), \
                    patch.object(run02.platform, 'system', return_value='Linux'), \
                    patch.object(run02, 'completed_training', return_value=False), \
                    patch.object(run02.subprocess, 'check_output', return_value='test==1\n'), \
                    patch.object(run02, 'run_logged', side_effect=[None, RuntimeError('merge failed')]) as child:
                with self.assertRaisesRegex(RuntimeError, 'merge failed'):
                    run02.main()
                self.assertEqual(child.call_count, 2)  # GPU check, then merge; no train.
                self.assertIn('merge', child.call_args_list[-1].args[0])


if __name__ == '__main__':
    unittest.main()
