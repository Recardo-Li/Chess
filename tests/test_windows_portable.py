"""Exercise a relocated Windows checkout with no user or Conda packages."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == "nt", "Windows runtime")
class PortableWindowsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.tmp = Path(temp.name)
        cls.clone = cls.tmp / "另一台电脑 Chess"
        cls.clone.mkdir()
        source = Path(__file__).resolve().parents[1]
        for name in ("run.py", "run.cmd"):
            shutil.copy2(source / name, cls.clone / name)
        for name in ("chess_eval", ".venv"):
            shutil.copytree(source / name, cls.clone / name, ignore=shutil.ignore_patterns("__pycache__"))
        config = cls.clone / "Conifg"
        config.mkdir()
        shutil.copyfile(source / "Conifg/Lichess.example.yaml", config / "Lichess.yaml")
        (config / "LLM.yaml").write_text(
            "provider: gpt\nmodel: test\nmax_plies: 10\nskill_path: rules.md\noutput_dir: outputs\n",
            encoding="utf-8",
        )
        (cls.clone / "rules.md").write_text("Choose a move.", encoding="utf-8")
        cls.scripts = cls.clone / ".venv" / "Scripts"
        system32 = Path(os.environ["SystemRoot"]) / "System32"
        cls.cmd = str(system32 / "cmd.exe")
        cls.env = {
            **os.environ,
            "PATH": str(system32),
            "PYTHONHOME": str(cls.tmp / "missing base"),
            "PYTHONPATH": str(cls.tmp / "unrelated modules"),
        }

    def invoke(self, command, *, env=None, cwd=None):
        return subprocess.run(
            command, cwd=cwd or self.tmp, env=env or self.env, stdin=subprocess.DEVNULL,
            capture_output=True, encoding="utf-8", errors="replace", timeout=30,
        )

    def test_bare_python_dispatches_from_an_interpreter_without_app_packages(self):
        # A second interpreter with only its standard library simulates Python on PATH.
        host = self.tmp / "system Python"
        shutil.copytree(self.scripts, host)
        (config,) = host.glob("python*._pth")
        (stdlib,) = host.glob("python*.zip")
        config.write_text(stdlib.name + "\n.\n", encoding="ascii")
        env = {**self.env, "PATH": str(host) + os.pathsep + self.env["PATH"]}
        clean = self.invoke([host / "python.exe", "-c",
                             "import importlib.util; assert importlib.util.find_spec('yaml') is None"],
                            env=env)
        self.assertEqual(clean.returncode, 0, clean.stderr)
        result = self.invoke([self.cmd, "/d", "/c", "python", "run.py", "--check-config"],
                             cwd=self.clone, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("检查通过", result.stdout)

    def test_command_launcher_without_python_on_path(self):
        command = [self.cmd, "/d", "/c", "call", str(self.clone / "run.cmd")]
        result = self.invoke([*command, "--check-config"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("检查通过", result.stdout)
        invalid = self.invoke([*command, "--unknown-option"])
        self.assertEqual(invalid.returncode, 2, invalid.stderr)
        self.assertIn("--unknown-option", invalid.stderr)

    def test_dependencies_and_https_certificates_resolve_inside_the_clone(self):
        probe = """\
import _ssl, certifi, chess, charset_normalizer, idna, json, requests, ssl, sys, urllib3, yaml
modules = (certifi, chess, charset_normalizer, idna, requests, urllib3, yaml)
print(json.dumps({
    "packages": [m.__file__ for m in modules] + [certifi.where()],
    "ssl": _ssl.__file__, "prefix": sys.base_prefix, "paths": sys.path,
    "roots": ssl.create_default_context(cafile=certifi.where()).cert_store_stats()["x509_ca"],
}))
"""
        result = self.invoke([self.scripts / "python.exe", "-c", probe])
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(result.stdout)
        runtime = self.clone / ".venv"
        for path in evidence["packages"]:
            self.assertTrue(Path(path).resolve().is_relative_to(runtime / "Lib" / "site-packages"), path)
        for path in (evidence["ssl"], evidence["prefix"]):
            self.assertTrue(Path(path).resolve().is_relative_to(runtime), path)
        for path in evidence["paths"]:
            self.assertTrue(Path(path).resolve().is_relative_to(self.clone), path)
        self.assertGreater(evidence["roots"], 0)
