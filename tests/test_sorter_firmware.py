"""Compile the actual .ino against a host clock/Serial/Servo harness; no hardware."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class FirmwareTests(unittest.TestCase):
    def test_protocol_and_timed_sequence_on_all_three_positions(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        if not compiler:
            self.skipTest('Host C++ compiler unavailable; install clang or g++')
        source = Path(__file__).resolve().parent/'firmware'
        with tempfile.TemporaryDirectory() as td:
            exe = str(Path(td)/'sequence')
            flags = []
            if sys.platform == 'darwin':
                sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
                flags = ['-isystem', str(Path(sdk)/'usr/include/c++/v1')]
            subprocess.run([compiler, *flags, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I', str(source),
                            str(source/'sequence.cpp'), '-o', exe], check=True, capture_output=True, text=True)
            subprocess.run([exe], check=True, capture_output=True, text=True, timeout=10)
