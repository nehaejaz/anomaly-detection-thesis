from omegaconf import DictConfig, ListConfig, OmegaConf
from pathlib import Path
from typing import Union

def get_configurable_parameters(
    config_path: Union[Path, str, None] = None,
    weight_file: Union[str, None] = None,
    config_filename: Union[str, None] = "config",
    config_file_extension: Union[str, None] = "yaml",
) -> Union[DictConfig, ListConfig]:
    """Get configurable parameters.

    Args:
        model_name: str | None:  (Default value = None)
        config_path: Path | str | None:  (Default value = None)
        weight_file: Path to the weight file
        config_filename: str | None:  (Default value = "config")
        config_file_extension: str | None:  (Default value = "yaml")

    Returns:
        DictConfig | ListConfig: Configurable parameters in DictConfig object.
    """
    # if model_name is None is config_path:
    #     raise ValueError(
    #         "Both model_name and model config path cannot be None! "
    #         "Please provide a model name or path to a config file!"
    #     )

    # if model_name == "efficientad":
    #     warnings.warn("`efficientad` is deprecated as --model. Please use `efficient_ad` instead.", DeprecationWarning)
    #     model_name = "efficient_ad"

    if config_path is None:
        raise ValueError(
                    "Please provide a config file path!"
                )
    config = OmegaConf.load(config_path)

    # keep track of the original config file because it will be modified
    config_original: DictConfig = config.copy()

    # if the seed value is 0, notify a user that the behavior of the seed value zero has been changed.
    if config.project.get("seed") == 0:
        warn(
            "The seed value is now fixed to 0. "
            "Up to v0.3.7, the seed was not fixed when the seed value was set to 0. "
            "If you want to use the random seed, please select `None` for the seed value "
            "(`null` in the YAML file) or remove the `seed` key from the YAML file."
        )

    return config
