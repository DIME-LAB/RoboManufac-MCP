"""Locate aruco-grasp-annotator data directory relative to repo or ~/Documents."""
from pathlib import Path
from typing import Optional


def _try_repo_relative_path() -> Optional[Path]:
    """
    Attempt to locate the repo root by walking up from this file and
    checking for the aruco-grasp-annotator directory.
    """
    current = Path(__file__).resolve()
    target_dir = "aruco-grasp-annotator"
    for parent in current.parents:
        candidate = parent / target_dir
        if candidate.is_dir():
            data_dir = candidate / "data"
            if (data_dir / "grasp").exists():
                return data_dir
    return None


def _search_documents() -> Optional[Path]:
    """Fallback search inside ~/Documents."""
    documents_dir = Path.home() / "Documents"
    if not documents_dir.exists():
        return None

    target_name = "aruco-grasp-annotator"
    for path in documents_dir.rglob(target_name):
        if path.is_dir():
            data_dir = path / "data"
            if data_dir.exists() and (data_dir / "grasp").exists():
                return data_dir
    return None


def find_aruco_data_dir() -> Optional[Path]:
    """
    Locate aruco-grasp-annotator/data directory.
    
    Returns:
        Path to data directory if found, None otherwise
    """
    return _try_repo_relative_path() or _search_documents()


def get_aruco_data_dir() -> Path:
    """
    Get aruco-grasp-annotator data directory.
    
    Returns:
        Path to data directory (raises FileNotFoundError if not found)
    """
    found_dir = find_aruco_data_dir()
    if found_dir:
        return found_dir
    
    raise FileNotFoundError(
        "Could not find aruco-grasp-annotator data directory relative to repo or in Documents folder."
    )


def get_symmetry_dir() -> Path:
    """Get symmetry directory path."""
    return get_aruco_data_dir() / "symmetry"


def get_assembly_data_dir() -> Path:
    """Get assembly data directory path (same as data dir)."""
    return get_aruco_data_dir()
