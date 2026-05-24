import logging
import re
import shutil
import tarfile
from itertools import product
from pathlib import Path

from huggingface_hub import snapshot_download

log = logging.getLogger(__name__)

HF_REPO_ID = "leggedrobotics/grand_tour_dataset"
HF_REVISION_MAIN = "main"
HF_REVISION_LIMO = (
    "refs/pr/6"  # PR that adds limo paths (teleop_paths, geometric_paths)
)

IMAGE_TOPICS = {"hdr_front", "hdr_left", "hdr_right"}


def pull_mission_topics(
    missions: list[str],
    topics: list[str],
    dataset_folder: Path,
    revision: str = HF_REVISION_MAIN,
    skip_existing: bool = True,
) -> Path:
    """Download specific zarr topics for the given missions from HuggingFace.

    Topics that already exist locally are skipped when skip_existing=True.
    Returns dataset_folder (unchanged).
    """
    dataset_folder = Path(dataset_folder)

    allow_patterns: list[str] = []
    for mission, topic in product(missions, topics):
        mission_dir = dataset_folder / mission
        if skip_existing and _topic_exists(mission_dir, topic):
            continue
        allow_patterns.append(f"{mission}/*{topic}*")

    if not allow_patterns:
        return dataset_folder

    hf_cache = snapshot_download(
        repo_id=HF_REPO_ID,
        revision=revision,
        allow_patterns=allow_patterns,
        repo_type="dataset",
    )
    _extract_to_folder(Path(hf_cache), dataset_folder, allow_patterns)
    return dataset_folder


def _topic_exists(mission_dir: Path, topic: str) -> bool:
    return (mission_dir / "data" / topic).exists() or (
        mission_dir / "images" / topic
    ).exists()


def _extract_to_folder(cache: Path, dest: Path, allow_patterns: list[str]) -> None:
    regex = _patterns_to_regex(allow_patterns)
    files = [f for f in cache.rglob("*") if regex.match(str(f.relative_to(cache)))]
    tar_files = [f for f in files if f.suffix == ".tar"]
    other_files = [f for f in files if f.suffix != ".tar" and f.is_file()]

    for src in tar_files:
        dst_parent = dest / src.relative_to(cache).parent
        dst_parent.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(src) as tar:
                tar.extractall(path=dst_parent)
        except Exception as exc:
            log.warning("Failed to extract %s: %s", src, exc)

    for src in other_files:
        dst = dest / src.relative_to(cache)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _patterns_to_regex(patterns: list[str]) -> re.Pattern:
    parts = []
    for p in patterns:
        p = re.escape(p).replace(r"\*", ".*").replace(r"\?", ".")
        parts.append(f"^{p}$")
    return re.compile("|".join(parts))
