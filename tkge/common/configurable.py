from typing import TypeVar, Any, Optional, Dict

from tkge.common.config import Config

import abc
from abc import ABCMeta, abstractmethod, abstractproperty

T = TypeVar("T", bound="Configurable")


class Configurable(metaclass=ABCMeta):
    """Mix-in class for adding configurations to objects.

    Each configured object that inherits from :class:`Configurable` has access to a `config` and a `configuration_key` that
    indicates where the object's options can be found in `config`.
    """

    def __init__(self, config: Config, configuration_key: str = None):
        self._init_configuration(config, configuration_key)


    # TODO(gengyuan) add load default
    # @property
    # @abstractmethod
    # def key_mapping(self):
    #     raise NotImplementedError
    #
    # def _parse_config(self):
    #     """Parsing :Class:`Config` instance to a new purified :Class:`Config` instance.
    #
    #     Args:
    #     config: Config instance with all needed params
    #     load_default: if true, load default parameters when absent; otherwise throw ConfigurationError
    #     key_mapping: mapping the old key in config instance to new key
    #     """
    #
    #     # TODO(gengyuan) option to load default
    #     # load_default = self.config.get("global.load_default")
    #
    #     for k, v in self.key_mapping.items():
    #         setattr(self, k, self.config.get(v))
    #
    #         print(k, self.__dict__[k])

    def _init_configuration(self, config: Config, configuration_key: Optional[str]):
        """Initializes `self.config` and `self.configuration_key`.

        Only after this method has been called, `get_option`, `check_option`, and
        `set_option` should be used. This method is automatically called in the
        constructor of this class, but can also be called by subclasses before calling
        the superclass constructor to allow access to these three methods. May also be
        overridden by subclasses to perform additional configuration.
        """
        self.config = config
        self.configuration_key = configuration_key

    def has_option(self, name: str) -> bool:
        """Returns True if an option for the specified key (name) exists in the configuration."""
        try:
            self.get_option(name)
            return True
        except KeyError:
            return False

    def get_option(self, name: str) -> Any:
        """Returns the value of the specified key (name)."""
        if self.configuration_key:
            # in case the configurable object is associated with a key in the config
            return self.config.get_default(self.configuration_key + "." + name)
        else:
            self.config.get(name)

    def check_option(self, name: str, allowed_values) -> Any:
        """Raises an error if value or default value of key (name) is not in allowed."""
        if self.configuration_key:
            # in case the configurable object is associated with a key in the config
            return self.config.check_default(
                self.configuration_key + "." + name, allowed_values
            )
        else:
            return self.config.check_default(name, allowed_values)

    def set_option(
            self, name: str, value, create=False, overwrite=Config.Overwrite.Yes, log=False
    ) -> Any:
        """Sets the value of the specified key (name)."""
        if self.configuration_key:
            # in case the configurable object is associated with a key in the config
            return self.config.set(
                self.configuration_key + "." + name,
                value,
                create=create,
                overwrite=overwrite,
                log=log,
            )
        else:
            return self.config.set(
                name, value, create=create, overwrite=overwrite, log=log
            )
