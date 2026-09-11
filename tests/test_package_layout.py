"""Offline regressions for stage-package imports and bundled model resources."""

import importlib
import importlib.resources
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class PackageLayoutTest(unittest.TestCase):
    def test_stage_package_imports_do_not_load_optional_dependencies(self):
        # A fresh interpreter proves package imports cannot accidentally depend
        # on providers already imported by another test in this process.
        script = textwrap.dedent('''
            import importlib
            import importlib.abc
            import sys

            blocked = {
                'torch', 'numpy', 'scipy', 'openai', 'google', 'websockets',
                'sounddevice', 'whisper', 'faster_whisper', 'transformers',
            }

            class RejectOptionalImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in blocked:
                        raise AssertionError('Package eagerly loaded ' + fullname)

            sys.meta_path.insert(0, RejectOptionalImports())
            for stage in ('getter', 'slicer', 'transcriber', 'translator',
                          'translator.legacy', 'exporter', 'utils'):
                importlib.import_module('stream_translator_gpt.' + stage)
            assert not blocked.intersection(sys.modules)
        ''')
        result = subprocess.run([sys.executable, '-B', '-S', '-c', script],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_silero_resource_is_located_beside_slicer_module(self):
        from stream_translator_gpt.slicer import audio_slicer

        resource = importlib.resources.files('stream_translator_gpt.slicer').joinpath('silero_vad.jit')
        self.assertTrue(resource.is_file())
        model_path = Path(audio_slicer.__file__).with_name('silero_vad.jit')
        self.assertTrue(model_path.is_file())
        self.assertEqual(resource.read_bytes(), model_path.read_bytes())

    def test_bundled_vad_loads_and_evaluates_without_network(self):
        import numpy as np
        import torch

        from stream_translator_gpt.slicer.audio_slicer import VAD

        grad_enabled = torch.is_grad_enabled()
        try:
            vad = VAD()
            probability = vad.get_speech_prob(np.zeros(512, dtype=np.float32))
            self.assertGreaterEqual(probability, 0)
            self.assertLessEqual(probability, 1)
        finally:
            torch.set_grad_enabled(grad_enabled)

    def test_vendor_tokenizer_and_mel_assets_still_resolve_from_module_paths(self):
        from stream_translator_gpt.transcriber.simul_streaming.simul_whisper.whisper import audio, tokenizer

        package = 'stream_translator_gpt.transcriber.simul_streaming.simul_whisper.whisper'
        assets = importlib.resources.files(package).joinpath('assets')
        for name in ('gpt2', 'multilingual'):
            self.assertTrue(assets.joinpath(f'{name}.tiktoken').is_file())
            encoding = tokenizer.get_encoding(name)
            original = 'こんにちは、配信テストです。'
            self.assertEqual(encoding.decode(encoding.encode(original)), original)
        self.assertTrue(assets.joinpath('mel_filters.npz').is_file())
        for n_mels in (80, 128):
            self.assertEqual(tuple(audio.mel_filters('cpu', n_mels).shape), (n_mels, 201))

    def test_vendor_license_resource_is_retained(self):
        resource = importlib.resources.files(
            'stream_translator_gpt.transcriber.simul_streaming').joinpath('LICENCE.txt')
        self.assertTrue(resource.is_file())
        self.assertGreater(len(resource.read_text(encoding='utf-8')), 100)

    def test_filter_modules_are_owned_by_transcription_stage(self):
        from stream_translator_gpt.transcriber import filters

        self.assertEqual(filters.__name__, 'stream_translator_gpt.transcriber.filters')
        for name in ('common', 'chinese', 'english', 'japanese'):
            module = importlib.import_module(f'stream_translator_gpt.transcriber.filters.{name}')
            self.assertEqual(Path(module.__file__).parent, Path(filters.__file__).parent)


if __name__ == '__main__':
    unittest.main()
