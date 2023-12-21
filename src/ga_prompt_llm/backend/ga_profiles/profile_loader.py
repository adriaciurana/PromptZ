import importlib.util
import os
import sys
from pathlib import Path

CURR_PATH = Path(__file__).parent / "../"
sys.path.append(str(CURR_PATH))

from default_profiles import PROFILES as DEFAULT_PROFILES

PROFILES = None
profiles_path = os.environ.get("GA_PROFILES_PATH", None)
if profiles_path is not None:
    profiles_path = Path(profiles_path)
    if profiles_path.exists():
        profiles_spec = importlib.util.spec_from_file_location(
            "profiles", profiles_path
        )
        profiles = importlib.util.module_from_spec(profiles_spec)
        profiles_spec.loader.exec_module(profiles)
        PROFILES = profiles.PROFILES

if PROFILES is None:
    PROFILES = DEFAULT_PROFILES


def load_profile(name: str):
    global PROFILES
    return PROFILES[name]


def profile_names() -> list[str]:
    return list(PROFILES.keys())
