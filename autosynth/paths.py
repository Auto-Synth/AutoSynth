from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def project_relative_string(path):
    path = resolve_project_path(path)
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def require_project_relative_path(path, *, label, source):
    if Path(path).is_absolute():
        raise ValueError(f"{source}: {label} must be relative to the project root")
    return resolve_project_path(path)
