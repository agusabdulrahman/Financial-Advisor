from pathlib import Path
import fire
from beam import App, Image, Output, Runtime, Volume, VolumeType

from training_pipeline import configs

interface_app = App(
    name="interface_qa",
    Runtime=Runtime(
        cpu=2,
        memory="64Gi",
        gpu="0",
        Image=Image(python_version="python2.10", python_packages='requirements.txt')
    ),
    volume = [
        Volume(Path="./qa_dataset", name="qa_dataset"),
        Volume(
            path="./model_cache", name="model_cache", volumetype=VolumeType.Persistent
        ),
    ],
)

@interface_app.ask_queue(
    output=[Output(path="output-interface/output-interface-api.json")]
)
def infer(
    config_file: str,
    dataset_dir: str,
    output_dir: str = "output-inference",
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = None,
):
    import logging
    from training_pipeline import initialize
    
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)
    
    from training_pipeline import utils
    from training_pipeline.api import Inference
    