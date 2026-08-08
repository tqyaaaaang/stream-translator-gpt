import os
import subprocess


def _force_utf8_environment(env: dict[str, str] | None = None) -> dict[str, str]:
    forced_env = (os.environ if env is None else env).copy()
    forced_env["PYTHONUTF8"] = "1"
    forced_env["PYTHONIOENCODING"] = "utf-8"
    return forced_env


def build_utf8_subprocess_env(project_root: str | None = None) -> dict[str, str]:
    """Return a child environment that makes Python text streams consistently UTF-8."""
    env = _force_utf8_environment()

    if project_root and os.path.isdir(os.path.join(project_root, "stream_translator_gpt")):
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (project_root + os.pathsep + existing_pythonpath
                             if existing_pythonpath else project_root)
    return env


def popen_utf8(args, **kwargs):
    """Start a text-mode subprocess whose pipes are decoded explicitly as UTF-8."""
    kwargs.update(text=True, encoding="utf-8", errors="replace", env=_force_utf8_environment(kwargs.get("env")))
    return subprocess.Popen(args, **kwargs)


def run_utf8(args, **kwargs):
    """Run a text-mode subprocess whose captured output is decoded explicitly as UTF-8."""
    kwargs.update(text=True, encoding="utf-8", errors="replace", env=_force_utf8_environment(kwargs.get("env")))
    return subprocess.run(args, **kwargs)
