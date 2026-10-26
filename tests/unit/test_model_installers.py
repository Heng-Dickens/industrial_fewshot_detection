import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import install_models as installer
from scripts.check_model_installation import asset_paths
from scripts import import_prompt_detectors as sources


class ModelInstallerTests(unittest.TestCase):
    def test_plan_never_runs_commands(self):
        with patch.object(installer.subprocess,'run',side_effect=AssertionError('Plan executed a command')):
            for backend in installer.BACKENDS:
                for extra in ([],['--all','--plan'],['--download-weights','--plan']):
                    with self.subTest(backend=backend,extra=extra),contextlib.redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(installer.main(['--backend',backend,*extra]),0)
                        self.assertIn('PLAN ONLY',output.getvalue())

    def test_pip_is_isolated_and_space_paths_are_single_arguments(self):
        root=Path('D:/workspace with spaces/project').resolve()
        for backend in installer.BACKENDS:
            args=installer.options(['--backend',backend,'--all','--python',sys.executable])
            plan=installer.build_plan(args,root)
            pip_steps=[s for s in plan if s.argv[1:4]==['-m','pip','install']]
            self.assertTrue(pip_steps)
            self.assertTrue(all(s.argv[0]==str(installer.environment_python(root,backend)) for s in pip_steps))
            venv=next(s for s in plan if s.kind=='environment')
            self.assertNotIn('--system-site-packages',venv.argv)
            self.assertEqual(Path(venv.argv[-1]),root/f'.venv-{backend}')

    def test_modes_select_only_their_source_and_assets(self):
        for mode in ('visual','text','fusion'):
            args=installer.options(['--backend','yoloe','--mode',mode,'--all'])
            plan=installer.build_plan(args)
            self.assertEqual(plan[0].argv[-2:],['--backend','yoloe'])
            text_download=[s for s in plan if any('download_yoloe_text_encoder.py' in x for x in s.argv)]
            self.assertEqual(bool(text_download),mode!='visual')
        args=installer.options(['--backend','wedetect','--variant','large','--all'])
        plan=installer.build_plan(args)
        self.assertTrue(any(s.argv[-2:]==['--variant','large'] for s in plan))
        self.assertFalse(any('mmcv' in arg or 'mmdet' in arg for s in plan for arg in s.argv))
        assets=asset_paths('wedetect','text','large')
        self.assertTrue(any('xlm-roberta-large' in str(p) for p in assets))
        self.assertTrue(any(p.name=='wedetect_large.pth' for p in assets))

    def test_pet_ops_wheel_and_numpy_constraints(self):
        args=installer.options(['--backend','pet_dino','--torch-build','cu121','--all'])
        step=next(s for s in installer.build_plan(args) if s.kind=='mmcv')
        self.assertIn('--only-binary=mmcv',step.argv)
        self.assertIn('--constraint',step.argv)
        self.assertIn('mmcv==2.1.0',step.argv)
        self.assertIn('https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html',step.argv)
        for argv in (['--backend','pet_dino','--torch-build','cu124'],['--backend','pet_dino','--mode','fusion'],
                     ['--backend','groundingdino','--variant','large'],['--backend','wedetect','--mode','visual']):
            with self.subTest(argv=argv),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                installer.options(argv)

    def test_failure_stops_later_steps_and_explains_mmcv(self):
        args=installer.options(['--backend','wedetect','--all'])
        with patch.object(installer.subprocess,'run',side_effect=[SimpleNamespace(stdout='[3,10]',returncode=0),
                                                                SimpleNamespace(returncode=7)]) as run:
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'later steps'):
                installer.main(['--backend','wedetect','--all'])
            self.assertEqual(run.call_count,2)
        step=installer.Step('MMCV',['fake','pip'],kind='mmcv')
        with patch.object(installer.subprocess,'run',return_value=SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(RuntimeError,'mmcv-lite is not a substitute'):
                installer.run_step(step,args)

    def test_repeated_setup_preserves_environment_and_working_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            python=installer.environment_python(root,'yoloe')
            python.parent.mkdir(parents=True)
            python.write_text('fixture')
            (python.parent.parent/'pyvenv.cfg').write_text('fixture')
            args=installer.options(['--backend','yoloe','--all'])
            with patch.object(installer.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(str(python.parent.parent)))) as run,contextlib.redirect_stdout(io.StringIO()):
                installer.run_step(installer.Step('env',[],kind='environment'),args,root)
                self.assertEqual(run.call_count,1)
            with patch.object(installer.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(str(root/'wrong')))):
                with self.assertRaisesRegex(RuntimeError,'another environment'):
                    installer.run_step(installer.Step('env',[],kind='environment'),args,root)
            with patch.object(installer.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='2.13.0')) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    installer.run_step(installer.Step('torch',['fake'],kind='torch'),args,root)
                self.assertEqual(run.call_count,1)

    def test_source_import_keeps_existing_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dest=root/'third_party/wedetect'
            entry=dest/sources.ENTRIES['wedetect']
            entry.parent.mkdir(parents=True)
            entry.write_text('# fixture')
            (dest/'UPSTREAM.json').write_text(json.dumps({'revision':sources.REPOS['wedetect'][1]}))
            with patch.object(sources,'ROOT',root),patch.object(sources.urllib.request,'urlopen',side_effect=AssertionError('No download')):
                with contextlib.redirect_stdout(io.StringIO()):
                    sources.main(['--backend','wedetect'])
                (dest/'UPSTREAM.json').write_text(json.dumps({'revision':'other'}))
                with self.assertRaisesRegex(RuntimeError,'refusing to overwrite'):
                    sources.main(['--backend','wedetect'])

    @unittest.skipUnless(shutil.which('powershell') or shutil.which('pwsh'),'PowerShell required')
    def test_powershell_entrypoints_forward_plan_options(self):
        shell=shutil.which('powershell') or shutil.which('pwsh')
        root=Path(__file__).resolve().parents[2]
        cases=[('setup_yoloe.ps1',['-Mode','fusion']),('setup_groundingdino.ps1',['-Variant','base']),
               ('setup_pet_dino.ps1',['-Mode','text']),('setup_wedetect.ps1',['-Variant','large']),
               ('download_groundingdino.ps1',['-Variant','base'])]
        for name,extra in cases:
            with self.subTest(name=name):
                result=subprocess.run([shell,'-NoProfile','-File',str(root/'scripts'/name),'-Python',sys.executable,
                                       *extra,'-Plan'],capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('PLAN ONLY',result.stdout)
                if 'base' in extra:
                    self.assertIn('--variant base',result.stdout)
                if 'large' in extra:
                    self.assertIn('--variant large',result.stdout)


if __name__=='__main__':
    unittest.main()
