import datetime
import os
import re
import sys
import time
import uuid
from enum import Enum
from functools import reduce
from typing import Any, Dict, Optional, Union, TypeVar, Tuple

import yaml

from tkge.common.error import ConfigurationError

T = TypeVar("T", bound="Config")


class Config:
    """Configuration class for all configurable classes.

    `Config` instance could be created from a configuration file (.yaml) or from a param dict.
    """

    Overwrite = Enum("Overwrite", "Yes No Error")

    def __init__(self, options: Dict[str, Any]):
        self.options = options

        self.root_folder = self.get("task.folder")  # main folder (config file, checkpoints, ...)

    def create_experiment(self):
        self.ex_folder, self.ex_id = self.create_exid(self.root_folder)

        self.save(os.path.join(self.ex_folder, 'config.yaml'))

        self.checkpoint_folder = os.path.join(self.ex_folder, 'ckpt')
        self.log_folder = os.path.join(self.ex_folder,
                                       'logging')  # None means use self.folder; used for kge.log, trace.yaml
        self.log_prefix: str = None

        os.makedirs(self.checkpoint_folder)
        os.makedirs(self.log_folder)

        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_file = os.path.join(self.log_folder, f"{self.start_time}.log")
        self.log_level = self.get("console.log_level")
        self.echo = self.get("console.echo")

    def restore_experiment(self, ex_folder):
        """
        ex_folder are absolute path
        """
        self.ex_folder = ex_folder

        self.checkpoint_folder = os.path.join(self.ex_folder, 'ckpt')
        self.log_folder = os.path.join(self.ex_folder, 'logging')
        self.log_prefix = None

        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        for f in os.listdir(self.log_folder):
            if re.match(r"\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}.log", f):
                self.log_file = os.path.join(self.log_folder, f)
                break
        else:
            self.start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.log_file = os.path.join(self.log_folder, f"{self.start_time}.log")

        self.log_level = self.get("console.log_level")
        self.echo = self.get("console.echo")

    def create_trial(self, trial_id: int):
        self.trial_id = trial_id
        self.trial_folder = os.path.join(self.ex_folder, 'trial' + str(trial_id))
        os.makedirs(self.trial_folder)

        self.save(os.path.join(self.trial_folder, 'config.yaml'))

        self.checkpoint_folder = os.path.join(self.trial_folder, 'ckpt')
        self.log_folder = os.path.join(self.trial_folder, 'logging')
        self.log_prefix: str = None

        os.makedirs(self.checkpoint_folder)
        os.makedirs(self.log_folder)

        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_file = os.path.join(self.log_folder, f"{self.start_time}.log")
        self.log_level = self.get("console.log_level")
        self.echo = self.get("console.echo")

    def open_experiment(self):
        raise NotImplementedError

    @classmethod
    def create_from_yaml(cls, filepath: str):
        with open(filepath, "r") as file:
            options: Dict[str, Any] = yaml.load(file, Loader=yaml.SafeLoader)

        return cls(options=options)

    @classmethod
    def create_from_dict(cls, options: Dict[str, Any]):
        return cls(options=options)

    @classmethod
    def create_from_parent(cls, parent_config: T, child_key: str):
        return cls.create_from_dict(parent_config.get(child_key))

    def __repr__(self):
        return yaml.dump(self.options, default_flow_style=False, sort_keys=False)

    # Access Methods
    def get(self, key: str) -> Any:
        """Obtain value of specified key.

        Nested dictionary values can be accessed via "." (e.g., "job.type").
        """
        result = self.options

        for name in key.split("."):
            try:
                result = result[name]
            except KeyError:
                raise KeyError(f"Error accessing {name} for key {key}")

        if isinstance(result, str) and re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', result):
            result = float(result)

        return result

    def set(
            self, key: str, value, create=False, overwrite=Overwrite.Yes, log=False
    ) -> Any:
        """Set value of specified key.

        Nested dictionary values can be accessed via "." (e.g., "job.type").

        If ``create`` is ``False`` , raises :class:`ValueError` when the key
        does not exist already; otherwise, the new key-value pair is inserted
        into the configuration.

        """
        create = True
        from tkge.common.misc import is_number

        splits = key.split(".")
        data = self.options

        # flatten path and see if it is valid to be set
        path = []
        for i in range(len(splits) - 1):
            if splits[i] in data:
                create = create  # or "+++" in data[splits[i]]
            else:
                if create:
                    data[splits[i]] = dict()
                else:
                    raise ConfigurationError(
                        (
                                "{} cannot be set because creation of "
                                + "{} is not permitted"
                        ).format(key, ".".join(splits[: (i + 1)]))
                    )
            path.append(splits[i])
            data = data[splits[i]]

        # check correctness of value
        try:
            current_value = data.get(splits[-1])
        except:
            raise ConfigurationError(
                "These config entries {} {} caused an error.".format(data, splits[-1])
            )

        if current_value is None:
            if not create:
                raise ConfigurationError("key {} not present and `create` is disabled".format(key))

            if isinstance(value, str) and is_number(value, int):
                value = int(value)
            elif isinstance(value, str) and is_number(value, float):
                value = float(value)
        else:
            if (
                    isinstance(value, str)
                    and isinstance(current_value, float)
                    and is_number(value, float)
            ):
                value = float(value)
            elif (
                    isinstance(value, str)
                    and isinstance(current_value, int)
                    and is_number(value, int)
            ):
                value = int(value)
            if type(value) != type(current_value):
                raise ConfigurationError(
                    "key {} has incorrect type (expected {}, found {})".format(
                        key, type(current_value), type(value)
                    )
                )
            if overwrite == Config.Overwrite.No:
                return current_value
            if overwrite == Config.Overwrite.Error and value != current_value:
                raise ConfigurationError("key {} cannot be overwritten".format(key))

        # all fine, set value
        data[splits[-1]] = value
        if log:
            self.log("Set {}={}".format(key, value))
        return value

    def set_all(
            self, new_options: Dict[str, Any], create=False, overwrite=Overwrite.Yes
    ):
        """Updates the configuration with new options and overwrites them for existing keys."""
        for key, value in Config.flatten(new_options).items():
            self.set(key, value, create, overwrite)

    @staticmethod
    def flatten(options: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a dictionary of flattened configuration options."""
        result = {}
        Config.__flatten(options, result)
        return result

    @staticmethod
    def __flatten(options: Dict[str, Any], result: Dict[str, Any], prefix=""):
        """Flattens a nested dictionary recursively by appending nested keys and separating them by '.'"""
        for key, value in options.items():
            fullkey = key if prefix == "" else prefix + "." + key
            if type(value) is dict:
                Config.__flatten(value, result, prefix=fullkey)
            else:
                result[fullkey] = value

    # Logging and Tracing
    def log(self, msg: str, level="info"):
        """Add a message to the default log file.

        Optionally also print on console. ``prefix`` is used to indent each
        output line.

        """
        if not os.path.exists(self.log_file):
            mode = "w"
        else:
            mode = "a"

        with open(self.log_file, mode) as file:
            for line in msg.splitlines():
                levels = {"debug": 0,
                          "info": 1,
                          "warning": 2,
                          "error": 3}
                line = f"{level.upper()}: {line}"
                if levels.get(level) >= levels.get(self.log_level):
                    if self.echo:
                        print(line)
                    file.write(f"{str(datetime.datetime.now())} {line}\n")

    def trace(
            self, echo=False, echo_prefix="", echo_flow=False, log=False, **kwargs
    ) -> Dict[str, Any]:
        """Write a set of key-value pairs to the trace file.

        The pairs are written as a single-line YAML record. Optionally, also
        echo to console and/or write to log file.

        And id and the current time is automatically added using key ``timestamp``.

        Returns the written k/v pairs.
        """
        kwargs["timestamp"] = time.time()
        kwargs["entry_id"] = str(uuid.uuid4())
        line = yaml.dump(kwargs, width=float("inf"), default_flow_style=True).strip()
        if echo or log:
            msg = yaml.dump(kwargs, default_flow_style=echo_flow)
            if log:
                self.log(msg, echo, echo_prefix)
            else:
                for line in msg.splitlines():
                    if echo_prefix:
                        line = echo_prefix + line
                        print(line)
        with open(self.tracefile(), "a") as file:
            file.write(line + "\n")
        return kwargs

    def assert_true(self, condition: bool, message: str):
        if not condition:
            self.log(message, level="error")
            sys.exit()

    # -- FOLDERS AND CHECKPOINTS ----------------------------------------------

    def checkpoint_file(self, cpt_id: Union[str, int]) -> str:
        """Returns path of checkpoint file for given checkpoint id"""
        from tkge.common.misc import is_number

        if is_number(cpt_id, int):
            return os.path.join(self.root_folder, "checkpoint_{:05d}.pt".format(int(cpt_id)))
        else:
            return os.path.join(self.root_folder, "checkpoint_{}.pt".format(cpt_id))

    def last_checkpoint(self) -> Optional[int]:
        """Returns epoch number of latest checkpoint"""
        # stupid implementation, but works
        tried_epoch = 0
        found_epoch = 0
        while tried_epoch < found_epoch + 500:
            tried_epoch += 1
            if os.path.exists(self.checkpoint_file(tried_epoch)):
                found_epoch = tried_epoch
        if found_epoch > 0:
            return found_epoch
        else:
            return None

    # @staticmethod
    # def get_best_or_last_checkpoint(path: str) -> str:
    #     """Returns best (if present) or last checkpoint path for a given folder path."""
    #     config = Config(folder=path, load_default=False)
    #     checkpoint_file = config.checkpoint_file("best")
    #     if os.path.isfile(checkpoint_file):
    #         return checkpoint_file
    #     cpt_epoch = config.last_checkpoint()
    #     if cpt_epoch:
    #         return config.checkpoint_file(cpt_epoch)
    #     else:
    #             raise Exception("Could not find checkpoint in {}".format(path))

    # -- CONVENIENCE METHODS --------------------------------------------------

    def _check(self, key: str, value, allowed_values) -> Any:
        if value not in allowed_values:
            raise ValueError(
                "Illegal value {} for key {}; allowed values are {}".format(
                    value, key, allowed_values
                )
            )
        return value

    def check(self, key: str, allowed_values) -> Any:
        """Raise an error if value of key is not in allowed.

        If fine, returns value.
        """
        return self._check(key, self.get(key), allowed_values)

    def check_range(
            self, key: str, min_value, max_value, min_inclusive=True, max_inclusive=True
    ) -> Any:
        value = self.get(key)
        if (
                value < min_value
                or (value == min_value and not min_inclusive)
                or value > max_value
                or (value == max_value and not max_inclusive)
        ):
            raise ValueError(
                "Illegal value {} for key {}; must be in range {}{},{}{}".format(
                    value,
                    key,
                    "[" if min_inclusive else "(",
                    min_value,
                    max_value,
                    "]" if max_inclusive else ")",
                )
            )
        return value

    def logdir(self) -> str:
        folder = self.log_folder if self.log_folder else self.ex_folder
        return folder

    def logfile(self) -> str:
        folder = self.log_folder if self.log_folder else self.ex_folder
        return os.path.join(folder, "kge.log")

    def tracefile(self) -> str:
        folder = self.log_folder if self.log_folder else self.ex_folder
        return os.path.join(folder, "trace.yaml")

    def create_exid(self, root_folder) -> Tuple[str, str]:
        """
        Get a self-incremental id in the current checkpoint folder.
        Every time when starting a new training task, a new experiment folder
        with incremental id will be created and associated with current experiment.
        """
        overall_ckpt_folder = root_folder
        base_folder = os.path.join(overall_ckpt_folder, self.get("model.type"), self.get("dataset.name"))

        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        ex_ls = os.listdir(base_folder)
        ex_max = int(reduce(lambda a, b: a if a > b else b, ex_ls)[2:]) if ex_ls else -1
        ex_id = f"ex{ex_max + 1:06d}"
        ex_folder = os.path.join(base_folder, ex_id)
        os.makedirs(ex_folder)

        return ex_folder, ex_id

    def save(self, filename):
        """Save this configuration to the given file"""
        with open(filename, "w+") as file:
            file.write(yaml.dump(self.options))


# class Config:
#     """
#     Configuration class for every experiment.
#
#     All available options, their types, and their descriptions are defined in
#     :file:`config_default.yaml`.
#     """
#
#     Overwrite = Enum("Overwrite", "Yes No Error")
#
#     def __init__(self, folder: str = None, load_default=True):
#         if load_default:
#             with open(folder, "r") as file:
#                 self.options: Dict[str, Any] = yaml.load(file, Loader=yaml.SafeLoader)
#
#         else:
#             with open(folder, "r") as file:
#                 self.options = yaml.load(file, Loader=yaml.SafeLoader)
#
#         self.folder = folder  # main folder (config file, checkpoints, ...)
#         self.log_folder = self.get("console.folder")  # None means use self.folder; used for kge.log, trace.yaml
#         self.log_prefix: str = None
#
#     def _import(self, module_name: str):
#         """Imports the specified module configuration.
#
#         Adds the configuration options from kge/model/<module_name>.yaml to
#         the configuration. Retains existing module configurations, but verifies
#         that fields and their types are correct.
#
#         """
#         import tkge.models.model, tkge.models.embedder
#         from tkge.common.misc import filename_in_module
#
#         # load the module_name
#         module_config = Config(load_default=False)
#         module_config.load(
#             filename_in_module(
#                 [tkge.models.model, tkge.models.embedder, ], "{}.yaml".format(module_name)
#             ),
#             create=True,
#         )
#         if "import" in module_config.options:
#             del module_config.options["import"]
#
#         # add/verify current configuration
#         for key in module_config.options.keys():
#             cur_value = None
#             try:
#                 cur_value = {key: self.get(key)}
#             except KeyError:
#                 continue
#             module_config.set_all(cur_value, create=False)
#
#         # now update this configuration
#         self.set_all(module_config.options, create=True)
#
#         # remember the import
#         imports = self.options.get("import")
#         if imports is None:
#             imports = module_name
#         elif isinstance(imports, str):
#             imports = [imports, module_name]
#         else:
#             imports.append(module_name)
#             imports = list(dict.fromkeys(imports))
#         self.options["import"] = imports
#
#     # Access Methods
#     def get(self, key: str, remove_plusplusplus=True) -> Any:
#         """Obtain value of specified key.
#
#         Nested dictionary values can be accessed via "." (e.g., "job.type"). Strips all
#         '+++' keys unless `remove_plusplusplus` is set to `False`.
#
#         """
#         result = self.options
#
#         for name in key.split("."):
#             try:
#                 result = result[name]
#             except KeyError:
#                 raise KeyError(f"Error accessing {name} for key {key}")
#
#         if isinstance(result, str) and re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', result):
#             result = float(result)
#
#         if remove_plusplusplus and isinstance(result, collections.Mapping):
#             def do_remove_plusplusplus(option):
#                 """Recursive function to remove '+++'"""
#                 if isinstance(option, collections.Mapping):
#                     option.pop("+++", None)
#                     for values in option.values():
#                         do_remove_plusplusplus(values)
#
#             # remove the '+++' not in the original options, but return a deepcopy with removed '+++'
#             result = copy.deepcopy(result)
#             do_remove_plusplusplus(result)
#
#         return result
#
#     def get_default(self, key: str) -> Any:
#         """Returns the value of the key if present or default if not.
#
#         The default value is looked up as follows. If the key has form ``parent.field``,
#         see if there is a ``parent.type`` property. If so, try to look up ``field``
#         under the key specified there (proceeds recursively). If not, go up until a
#         `type` field is found, and then continue from there.
#
#         """
#         try:
#             return self.get(key)
#         except KeyError as e:
#             last_dot_index = key.rfind(".")
#             if last_dot_index < 0:
#                 raise e
#             parent = key[:last_dot_index]
#             field = key[last_dot_index + 1:]
#             while True:
#                 # self.log("Looking up {}/{}".format(parent, field))
#                 try:
#                     parent_type = self.get(parent + "." + "type")
#                     # found a type -> go to this type and lookup there
#                     new_key = parent_type + "." + field
#                     last_dot_index = new_key.rfind(".")
#                     parent = new_key[:last_dot_index]
#                     field = new_key[last_dot_index + 1:]
#                 except KeyError:
#                     # no type found -> go up hierarchy
#                     last_dot_index = parent.rfind(".")
#                     if last_dot_index < 0:
#                         raise e
#                     field = parent[last_dot_index + 1:] + "." + field
#                     parent = parent[:last_dot_index]
#                     continue
#                 try:
#                     value = self.get(parent + "." + field)
#                     # uncomment this to see where defaults are taken from
#                     # self.log(
#                     #     "Using value of {}={} for key {}".format(
#                     #         parent + "." + field, value, key
#                     #     )
#                     # )
#                     return value
#                 except KeyError:
#                     # try further
#                     continue
#
#     def get_first_present_key(self, *keys: str, use_get_default=False) -> str:
#         """Return the first key for which ``get`` or ``get_default`` finds a value."""
#         for key in keys:
#             try:
#                 self.get_default(key) if use_get_default else self.get(key)
#                 return key
#             except KeyError:
#                 pass
#         raise KeyError("None of the following keys found: ".format(keys))
#
#     def set(
#             self, key: str, value, create=False, overwrite=Overwrite.Yes, log=False
#     ) -> Any:
#         """Set value of specified key.
#
#         Nested dictionary values can be accessed via "." (e.g., "job.type").
#
#         If ``create`` is ``False`` , raises :class:`ValueError` when the key
#         does not exist already; otherwise, the new key-value pair is inserted
#         into the configuration.
#
#         """
#         create = True
#         from tkge.common.misc import is_number
#
#         splits = key.split(".")
#         data = self.options
#
#         # flatten path and see if it is valid to be set
#         path = []
#         for i in range(len(splits) - 1):
#             if splits[i] in data:
#                 create = create or "+++" in data[splits[i]]
#             else:
#                 if create:
#                     data[splits[i]] = dict()
#                 else:
#                     raise ConfigurationError(
#                         (
#                                 "{} cannot be set because creation of "
#                                 + "{} is not permitted"
#                         ).format(key, ".".join(splits[: (i + 1)]))
#                     )
#             path.append(splits[i])
#             data = data[splits[i]]
#
#         # check correctness of value
#         try:
#             current_value = data.get(splits[-1])
#         except:
#             raise ConfigurationError(
#                 "These config entries {} {} caused an error.".format(data, splits[-1])
#             )
#
#         if current_value is None:
#             if not create:
#                 raise ConfigurationError("key {} not present and `create` is disabled".format(key))
#
#             if isinstance(value, str) and is_number(value, int):
#                 value = int(value)
#             elif isinstance(value, str) and is_number(value, float):
#                 value = float(value)
#         else:
#             if (
#                     isinstance(value, str)
#                     and isinstance(current_value, float)
#                     and is_number(value, float)
#             ):
#                 value = float(value)
#             elif (
#                     isinstance(value, str)
#                     and isinstance(current_value, int)
#                     and is_number(value, int)
#             ):
#                 value = int(value)
#             if type(value) != type(current_value):
#                 raise ConfigurationError(
#                     "key {} has incorrect type (expected {}, found {})".format(
#                         key, type(current_value), type(value)
#                     )
#                 )
#             if overwrite == Config.Overwrite.No:
#                 return current_value
#             if overwrite == Config.Overwrite.Error and value != current_value:
#                 raise ConfigurationError("key {} cannot be overwritten".format(key))
#
#         # all fine, set value
#         data[splits[-1]] = value
#         if log:
#             self.log("Set {}={}".format(key, value))
#         return value
#
#     def set_all(
#             self, new_options: Dict[str, Any], create=False, overwrite=Overwrite.Yes
#     ):
#         """Updates the configuration with new options and overwrites them for existing keys."""
#         for key, value in Config.flatten(new_options).items():
#             self.set(key, value, create, overwrite)
#
#     def load(
#             self,
#             filename: str,
#             create=False,
#             overwrite=Overwrite.Yes,
#             allow_deprecated=True,
#     ):
#         """Update configuration options from the specified YAML file.
#
#         All options that do not occur in the specified file are retained.
#
#         If ``create`` is ``False``, raises :class:`ValueError` when the file
#         contains a non-existing options. When ``create`` is ``True``, allows
#         to add options that are not present in this configuration.
#
#         If the file has an import or model field, the corresponding
#         configuration files are imported.
#
#         """
#         with open(filename, "r") as file:
#             new_options = yaml.load(file, Loader=yaml.SafeLoader)
#         self.load_options(
#             new_options,
#             create=create,
#             overwrite=overwrite,
#             allow_deprecated=allow_deprecated,
#         )
#
#     def load_options(
#             self, new_options, create=False, overwrite=Overwrite.Yes, allow_deprecated=True,
#     ):
#         """Like `load`, but loads from an options object obtained from `yaml.load`."""
#         # import model configurations
#         if "model" in new_options:
#             model = new_options.get("model")
#             # TODO not sure why this can be empty when resuming an ax
#             # search with model as a search parameter
#             if model:
#                 self._import(model)
#         if "import" in new_options:
#             imports = new_options.get("import")
#             if not isinstance(imports, list):
#                 imports = [imports]
#             for module_name in imports:
#                 self._import(module_name)
#             del new_options["import"]
#
#         # process deprecated options
#         if allow_deprecated:
#             new_options = _process_deprecated_options(Config.flatten(new_options))
#
#         # now set all options
#         self.set_all(new_options, create, overwrite)
#

#
#     @staticmethod
#     def flatten(options: Dict[str, Any]) -> Dict[str, Any]:
#         """Returns a dictionary of flattened configuration options."""
#         result = {}
#         Config.__flatten(options, result)
#         return result
#
#     @staticmethod
#     def __flatten(options: Dict[str, Any], result: Dict[str, Any], prefix=""):
#         """Flattens a nested dictionary recursively by appending nested keys and separating them by '.'"""
#         for key, value in options.items():
#             fullkey = key if prefix == "" else prefix + "." + key
#             if type(value) is dict:
#                 Config.__flatten(value, result, prefix=fullkey)
#             else:
#                 result[fullkey] = value
#
#     def clone(self, subfolder: str = None) -> T:
#         """Return a deep copy"""
#         new_config = Config(folder=copy.deepcopy(self.folder), load_default=False)
#         new_config.options = copy.deepcopy(self.options)
#         if subfolder is not None:
#             new_config.folder = os.path.join(self.folder, subfolder)
#         return new_config
#
#     # Logging and Tracing
#     def log(self, msg: str, echo=True, prefix=""):
#         """Add a message to the default log file.
#
#         Optionally also print on console. ``prefix`` is used to indent each
#         output line.
#
#         """
#         if not os.path.exists(self.logdir()):
#             os.makedirs(self.logdir(), 0o700)
#
#         with open(self.logfile(), "a") as file:
#             for line in msg.splitlines():
#                 if prefix:
#                     line = prefix + line
#                 if self.log_prefix:
#                     line = self.log_prefix + line
#                 if echo:
#                     print(line)
#                 file.write(str(datetime.datetime.now()) + " " + line + "\n")
#
#     def trace(
#             self, echo=False, echo_prefix="", echo_flow=False, log=False, **kwargs
#     ) -> Dict[str, Any]:
#         """Write a set of key-value pairs to the trace file.
#
#         The pairs are written as a single-line YAML record. Optionally, also
#         echo to console and/or write to log file.
#
#         And id and the current time is automatically added using key ``timestamp``.
#
#         Returns the written k/v pairs.
#         """
#         kwargs["timestamp"] = time.time()
#         kwargs["entry_id"] = str(uuid.uuid4())
#         line = yaml.dump(kwargs, width=float("inf"), default_flow_style=True).strip()
#         if echo or log:
#             msg = yaml.dump(kwargs, default_flow_style=echo_flow)
#             if log:
#                 self.log(msg, echo, echo_prefix)
#             else:
#                 for line in msg.splitlines():
#                     if echo_prefix:
#                         line = echo_prefix + line
#                         print(line)
#         with open(self.tracefile(), "a") as file:
#             file.write(line + "\n")
#         return kwargs
#
#     # -- FOLDERS AND CHECKPOINTS ----------------------------------------------
#
#     def init_folder(self):
#         """Initialize the output folder.
#
#         If the folder does not exists, create it, dump the configuration
#         there and return ``True``. Else do nothing and return ``False``.
#
#         """
#         if not os.path.exists(self.folder):
#             os.makedirs(self.folder)
#             os.makedirs(os.path.join(self.folder, "config"))
#             self.save(os.path.join(self.folder, "config.yaml"))
#             return True
#         return False
#
#     def checkpoint_file(self, cpt_id: Union[str, int]) -> str:
#         """Returns path of checkpoint file for given checkpoint id"""
#         from tkge.common.misc import is_number
#
#         if is_number(cpt_id, int):
#             return os.path.join(self.folder, "checkpoint_{:05d}.pt".format(int(cpt_id)))
#         else:
#             return os.path.join(self.folder, "checkpoint_{}.pt".format(cpt_id))
#
#     def last_checkpoint(self) -> Optional[int]:
#         """Returns epoch number of latest checkpoint"""
#         # stupid implementation, but works
#         tried_epoch = 0
#         found_epoch = 0
#         while tried_epoch < found_epoch + 500:
#             tried_epoch += 1
#             if os.path.exists(self.checkpoint_file(tried_epoch)):
#                 found_epoch = tried_epoch
#         if found_epoch > 0:
#             return found_epoch
#         else:
#             return None
#
#     @staticmethod
#     def get_best_or_last_checkpoint(path: str) -> str:
#         """Returns best (if present) or last checkpoint path for a given folder path."""
#         config = Config(folder=path, load_default=False)
#         checkpoint_file = config.checkpoint_file("best")
#         if os.path.isfile(checkpoint_file):
#             return checkpoint_file
#         cpt_epoch = config.last_checkpoint()
#         if cpt_epoch:
#             return config.checkpoint_file(cpt_epoch)
#         else:
#             raise Exception("Could not find checkpoint in {}".format(path))
#
#     # -- CONVENIENCE METHODS --------------------------------------------------
#
#     def _check(self, key: str, value, allowed_values) -> Any:
#         if value not in allowed_values:
#             raise ValueError(
#                 "Illegal value {} for key {}; allowed values are {}".format(
#                     value, key, allowed_values
#                 )
#             )
#         return value
#
#     def check(self, key: str, allowed_values) -> Any:
#         """Raise an error if value of key is not in allowed.
#
#         If fine, returns value.
#         """
#         return self._check(key, self.get(key), allowed_values)
#
#     def check_default(self, key: str, allowed_values) -> Any:
#         """Raise an error if value or default value of key is not in allowed.
#
#         If fine, returns value.
#         """
#         return self._check(key, self.get_default(key), allowed_values)
#
#     def check_range(
#             self, key: str, min_value, max_value, min_inclusive=True, max_inclusive=True
#     ) -> Any:
#         value = self.get(key)
#         if (
#                 value < min_value
#                 or (value == min_value and not min_inclusive)
#                 or value > max_value
#                 or (value == max_value and not max_inclusive)
#         ):
#             raise ValueError(
#                 "Illegal value {} for key {}; must be in range {}{},{}{}".format(
#                     value,
#                     key,
#                     "[" if min_inclusive else "(",
#                     min_value,
#                     max_value,
#                     "]" if max_inclusive else ")",
#                 )
#             )
#         return value
#
#     def logdir(self) -> str:
#         folder = self.log_folder if self.log_folder else self.folder
#         return folder
#
#     def logfile(self) -> str:
#         folder = self.log_folder if self.log_folder else self.folder
#         return os.path.join(folder, "kge.log")
#
#     def tracefile(self) -> str:
#         folder = self.log_folder if self.log_folder else self.folder
#         return os.path.join(folder, "trace.yaml")


def _process_deprecated_options(options: Dict[str, Any]):
    import re

    # renames given key (but not subkeys!)
    def rename_key(old_key, new_key):
        if old_key in options:
            print(
                "Warning: key {} is deprecated; use {} instead".format(
                    old_key, new_key
                ),
                file=sys.stderr,
            )
            if new_key in options:
                raise ValueError(
                    "keys {} and {} must not both be set".format(old_key, new_key)
                )
            value = options[old_key]
            del options[old_key]
            options[new_key] = value
            return True
        return False

    # renames a value
    def rename_value(key, old_value, new_value):
        if key in options and options.get(key) == old_value:
            print(
                "Warning: {}={} is deprecated; use {} instead".format(
                    key, old_value, new_value
                ),
                file=sys.stderr,
            )
            options[key] = new_value
            return True
        return False

    # renames a set of keys matching a regular expression
    def rename_keys_re(key_regex, replacement):
        renamed_keys = set()
        regex = re.compile(key_regex)
        for old_key in options.keys():
            new_key = regex.sub(replacement, old_key)
            if old_key != new_key:
                rename_key(old_key, new_key)
                renamed_keys.add(new_key)
        return renamed_keys

    # renames a value of keys matching a regular expression
    def rename_value_re(key_regex, old_value, new_value):
        renamed_keys = set()
        regex = re.compile(key_regex)
        for key in options.keys():
            if regex.match(key):
                if rename_value(key, old_value, new_value):
                    renamed_keys.add(key)
        return renamed_keys

    # 31.01.2020
    rename_key("negative_sampling.num_samples_s", "negative_sampling.num_samples.s")
    rename_key("negative_sampling.num_samples_p", "negative_sampling.num_samples.p")
    rename_key("negative_sampling.num_samples_o", "negative_sampling.num_samples.o")

    # 10.01.2020
    rename_key("negative_sampling.filter_positives_s", "negative_sampling.filtering.s")
    rename_key("negative_sampling.filter_positives_p", "negative_sampling.filtering.p")
    rename_key("negative_sampling.filter_positives_o", "negative_sampling.filtering.o")

    # 20.12.2019
    for split in ["train", "valid", "test"]:
        old_key = f"dataset.{split}"
        if old_key in options:
            rename_key(old_key, f"dataset.files.{split}.filename")
            options[f"dataset.files.{split}.type"] = "triples"
    for obj in ["entity", "relation"]:
        old_key = f"dataset.{obj}_map"
        if old_key in options:
            rename_key(old_key, f"dataset.files.{obj}_ids.filename")
            options[f"dataset.files.{obj}_ids.type"] = "map"

    # 14.12.2019
    rename_key(
        "negative_sampling.filter_true_s", "negative_sampling.filtering.s"
    )
    rename_key(
        "negative_sampling.filter_true_p", "negative_sampling.filtering.p"
    )
    rename_key(
        "negative_sampling.filter_true_o", "negative_sampling.filtering.o"
    )

    # 14.12.2019
    rename_key("negative_sampling.num_negatives_s", "negative_sampling.num_samples.s")
    rename_key("negative_sampling.num_negatives_p", "negative_sampling.num_samples.p")
    rename_key("negative_sampling.num_negatives_o", "negative_sampling.num_samples.o")

    # 30.10.2019
    rename_value("train.loss", "ce", "kl")
    rename_keys_re(r"\.regularize_args\.weight$", ".regularize_weight")
    for p in [1, 2, 3]:
        for key in rename_value_re(r".*\.regularize$", f"l{p}", "lp"):
            new_key = re.sub(r"\.regularize$", ".regularize_args.p", key)
            options[new_key] = p
            print(f"Set {new_key}={p}.", file=sys.stderr)

    # 21.10.2019
    rename_key("negative_sampling.score_func_type", "negative_sampling.implementation")

    # 1.10.2019
    rename_value("train.type", "1toN", "KvsAll")
    rename_value("train.type", "spo", "1vsAll")
    rename_keys_re(r"^1toN\.", "KvsAll.")
    rename_key("checkpoint.every", "train.checkpoint.every")
    rename_key("checkpoint.keep", "train.checkpoint.keep")
    rename_value("model", "inverse_relations_model", "reciprocal_relations_model")
    rename_keys_re(r"^inverse_relations_model\.", "reciprocal_relations_model.")

    # 30.9.2019
    rename_key("eval.metrics_per_relation_type", "eval.metrics_per.relation_type")
    rename_key("eval.metrics_per_head_and_tail", "eval.metrics_per.head_and_tail")
    rename_key(
        "eval.metric_per_argument_frequency_perc", "eval.metrics_per.argument_frequency"
    )
    return options
