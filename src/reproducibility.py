# src/reproducibility.py
"""Reproducibility manifest generation for LLM reasoning evaluation experiments.

Captures all information needed to reproduce an experiment run:
- Package versions
- Random seeds
- Configuration settings
- System information
- Git commit hash (if available)

Requirements: 8.2 - Generate a reproducibility manifest with exact versions and seeds
"""
import hashlib
import json
import os
import platform
import random
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import MODELS, RANDOM_SEED, REPO_ROOT, RETRY_CONFIG, STRATEGIES


# ------------------- Data Classes -------------------

@dataclass
class PackageVersion:
    """Represents a Python package and its version."""
    name: str
    version: str


@dataclass
class SystemInfo:
    """System information for reproducibility."""
    python_version: str
    platform: str
    platform_version: str
    machine: str
    processor: str


@dataclass
class GitInfo:
    """Git repository information."""
    commit_hash: Optional[str] = None
    branch: Optional[str] = None
    is_dirty: bool = False
    remote_url: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Configuration used for the experiment."""
    models: Dict[str, Dict[str, Any]]
    strategies: Dict[str, Dict[str, Any]]
    retry_config: Dict[str, Any]
    random_seed: int
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReproducibilityManifest:
    """Complete reproducibility manifest for an experiment run."""
    manifest_id: str
    experiment_id: str
    created_at: str
    system_info: SystemInfo
    git_info: GitInfo
    packages: List[PackageVersion]
    config: ExperimentConfig
    dataset_hash: Optional[str] = None
    dataset_path: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary for JSON serialization."""
        return {
            "manifest_id": self.manifest_id,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "system_info": asdict(self.system_info),
            "git_info": asdict(self.git_info),
            "packages": [asdict(p) for p in self.packages],
            "config": asdict(self.config),
            "dataset_hash": self.dataset_hash,
            "dataset_path": self.dataset_path,
            "notes": self.notes,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReproducibilityManifest":
        """Create manifest from dictionary."""
        return cls(
            manifest_id=data["manifest_id"],
            experiment_id=data["experiment_id"],
            created_at=data["created_at"],
            system_info=SystemInfo(**data["system_info"]),
            git_info=GitInfo(**data["git_info"]),
            packages=[PackageVersion(**p) for p in data["packages"]],
            config=ExperimentConfig(**data["config"]),
            dataset_hash=data.get("dataset_hash"),
            dataset_path=data.get("dataset_path"),
            notes=data.get("notes", ""),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "ReproducibilityManifest":
        """Deserialize manifest from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


# ------------------- Helper Functions -------------------

def get_system_info() -> SystemInfo:
    """Collect system information."""
    return SystemInfo(
        python_version=sys.version,
        platform=platform.system(),
        platform_version=platform.version(),
        machine=platform.machine(),
        processor=platform.processor(),
    )


def get_git_info(repo_path: Optional[Path] = None) -> GitInfo:
    """Get git repository information if available.
    
    Args:
        repo_path: Path to the repository root. Defaults to REPO_ROOT.
        
    Returns:
        GitInfo with commit hash, branch, dirty status, and remote URL.
    """
    if repo_path is None:
        repo_path = REPO_ROOT
    
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        return GitInfo()
    
    git_info = GitInfo()
    
    try:
        # Get current commit hash
        head_file = git_dir / "HEAD"
        if head_file.exists():
            head_content = head_file.read_text().strip()
            
            if head_content.startswith("ref: "):
                # HEAD points to a branch
                ref_path = head_content[5:]  # Remove "ref: " prefix
                git_info.branch = ref_path.split("/")[-1]
                
                # Get commit hash from the ref
                ref_file = git_dir / ref_path
                if ref_file.exists():
                    git_info.commit_hash = ref_file.read_text().strip()
            else:
                # Detached HEAD - content is the commit hash
                git_info.commit_hash = head_content
        
        # Check if working directory is dirty (has uncommitted changes)
        # This is a simplified check - just looks for index changes
        index_file = git_dir / "index"
        if index_file.exists():
            # We can't easily check for dirty state without git command
            # Set to False as default - proper check would require git command
            git_info.is_dirty = False
        
        # Get remote URL
        config_file = git_dir / "config"
        if config_file.exists():
            config_content = config_file.read_text()
            for line in config_content.split("\n"):
                if "url = " in line:
                    git_info.remote_url = line.split("url = ")[1].strip()
                    break
                    
    except Exception:
        # If anything fails, return partial info
        pass
    
    return git_info


def get_installed_packages() -> List[PackageVersion]:
    """Get list of installed Python packages and their versions.
    
    Returns:
        List of PackageVersion objects for key packages.
    """
    packages = []
    
    # Key packages for this project
    key_packages = [
        "pandas",
        "numpy",
        "openai",
        "anthropic",
        "google-generativeai",
        "requests",
        "matplotlib",
        "seaborn",
        "scipy",
        "tqdm",
        "hypothesis",
        "pytest",
        "tenacity",
    ]
    
    try:
        import importlib.metadata as metadata
        
        for pkg_name in key_packages:
            try:
                version = metadata.version(pkg_name)
                packages.append(PackageVersion(name=pkg_name, version=version))
            except metadata.PackageNotFoundError:
                packages.append(PackageVersion(name=pkg_name, version="not installed"))
                
    except ImportError:
        # Fallback for older Python versions
        try:
            import pkg_resources
            
            for pkg_name in key_packages:
                try:
                    version = pkg_resources.get_distribution(pkg_name).version
                    packages.append(PackageVersion(name=pkg_name, version=version))
                except pkg_resources.DistributionNotFound:
                    packages.append(PackageVersion(name=pkg_name, version="not installed"))
        except ImportError:
            pass
    
    return packages


def compute_file_hash(file_path: Path) -> Optional[str]:
    """Compute SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file to hash.
        
    Returns:
        Hex string of SHA-256 hash, or None if file doesn't exist.
    """
    if not file_path.exists():
        return None
    
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


# ------------------- Main Functions -------------------

def generate_manifest(
    experiment_id: Optional[str] = None,
    dataset_path: Optional[Path] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> ReproducibilityManifest:
    """Generate a reproducibility manifest for an experiment run.
    
    Captures all information needed to reproduce the experiment:
    - System information (Python version, platform, etc.)
    - Git repository state (commit hash, branch, dirty status)
    - Installed package versions
    - Configuration settings (models, strategies, seeds)
    - Dataset hash for verification
    
    Args:
        experiment_id: Unique identifier for the experiment. Auto-generated if None.
        dataset_path: Path to the dataset file for hash computation.
        custom_config: Additional configuration to include in the manifest.
        notes: Optional notes about the experiment.
        
    Returns:
        ReproducibilityManifest containing all reproducibility information.
    """
    # Generate IDs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    
    if experiment_id is None:
        experiment_id = f"exp_{timestamp}_{short_uuid}"
    
    manifest_id = f"manifest_{timestamp}_{short_uuid}"
    
    # Collect system info
    system_info = get_system_info()
    
    # Collect git info
    git_info = get_git_info()
    
    # Collect package versions
    packages = get_installed_packages()
    
    # Build experiment config
    config = ExperimentConfig(
        models=MODELS,
        strategies=STRATEGIES,
        retry_config=RETRY_CONFIG,
        random_seed=RANDOM_SEED,
        custom_config=custom_config or {},
    )
    
    # Compute dataset hash if path provided
    dataset_hash = None
    dataset_path_str = None
    if dataset_path is not None:
        dataset_path = Path(dataset_path)
        dataset_hash = compute_file_hash(dataset_path)
        dataset_path_str = str(dataset_path)
    
    # Create manifest
    manifest = ReproducibilityManifest(
        manifest_id=manifest_id,
        experiment_id=experiment_id,
        created_at=datetime.now().isoformat(),
        system_info=system_info,
        git_info=git_info,
        packages=packages,
        config=config,
        dataset_hash=dataset_hash,
        dataset_path=dataset_path_str,
        notes=notes,
    )
    
    return manifest


def save_manifest(
    manifest: ReproducibilityManifest,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    """Save a reproducibility manifest to a JSON file.
    
    Args:
        manifest: The manifest to save.
        output_dir: Directory to save the manifest. Defaults to results/manifests/.
        filename: Custom filename. Defaults to manifest_{manifest_id}.json.
        
    Returns:
        Path to the saved manifest file.
    """
    if output_dir is None:
        output_dir = REPO_ROOT / "results" / "manifests"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"manifest_{manifest.experiment_id}.json"
    
    output_path = output_dir / filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(manifest.to_json())
    
    return output_path


def load_manifest(manifest_path: Path) -> ReproducibilityManifest:
    """Load a reproducibility manifest from a JSON file.
    
    Args:
        manifest_path: Path to the manifest JSON file.
        
    Returns:
        ReproducibilityManifest loaded from the file.
        
    Raises:
        FileNotFoundError: If the manifest file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    manifest_path = Path(manifest_path)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        json_str = f.read()
    
    return ReproducibilityManifest.from_json(json_str)


def verify_manifest(manifest: ReproducibilityManifest) -> Dict[str, Any]:
    """Verify that current environment matches a manifest.
    
    Checks:
    - Python version matches
    - Key package versions match
    - Dataset hash matches (if dataset path provided)
    
    Args:
        manifest: The manifest to verify against.
        
    Returns:
        Dictionary with verification results:
        - matches: bool indicating if all checks pass
        - python_match: bool for Python version
        - package_mismatches: list of packages with different versions
        - dataset_match: bool for dataset hash (None if not applicable)
    """
    results = {
        "matches": True,
        "python_match": True,
        "package_mismatches": [],
        "dataset_match": None,
    }
    
    # Check Python version (major.minor only)
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    manifest_python = manifest.system_info.python_version.split()[0]
    manifest_python_short = ".".join(manifest_python.split(".")[:2])
    
    if current_python != manifest_python_short:
        results["python_match"] = False
        results["matches"] = False
    
    # Check package versions
    current_packages = {p.name: p.version for p in get_installed_packages()}
    
    for pkg in manifest.packages:
        if pkg.name in current_packages:
            if current_packages[pkg.name] != pkg.version:
                results["package_mismatches"].append({
                    "package": pkg.name,
                    "manifest_version": pkg.version,
                    "current_version": current_packages[pkg.name],
                })
                results["matches"] = False
    
    # Check dataset hash if applicable
    if manifest.dataset_path and manifest.dataset_hash:
        dataset_path = Path(manifest.dataset_path)
        if dataset_path.exists():
            current_hash = compute_file_hash(dataset_path)
            results["dataset_match"] = current_hash == manifest.dataset_hash
            if not results["dataset_match"]:
                results["matches"] = False
        else:
            results["dataset_match"] = False
            results["matches"] = False
    
    return results


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy (if available)
    - Environment variable PYTHONHASHSEED
    
    Args:
        seed: The seed value to use.
    """
    # Python random
    random.seed(seed)
    
    # Environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
