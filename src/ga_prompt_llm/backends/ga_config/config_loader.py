import importlib.util
import os
import sys
from pathlib import Path

CURR_PATH = Path(__file__).parent / "../"
sys.path.append(str(CURR_PATH))

from default_configs import CONFIGS as DEFAULT_CONFIGS

# os.environ[
#     "GA_CONFIG_PATH"
# ] = "/home/totolia/GA-prompt-LLM/src/ga_prompt_llm/backends/default_configs.py"

CONFIGS = None
config_path = os.environ.get("GA_CONFIG_PATH", None)
if config_path is not None:
    config_path = Path(config_path)
    if config_path.exists():
        config_spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config)
        CONFIGS = config.CONFIGS

if CONFIGS is None:
    CONFIGS = DEFAULT_CONFIGS


def load_config(name: str):
    global CONFIGS
    return CONFIGS[name]
