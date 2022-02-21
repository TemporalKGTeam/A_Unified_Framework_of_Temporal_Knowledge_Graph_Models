class ConfigurationError(Exception):
    """Exception for error cases in the `tkge.common.config.Config` class."""

    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg

    def __str__(self):
        return self.msg


class CodeError(Exception):
    """Exception for non-standard or incompatible code"""

    def __init__(self, msg: str):
        super().__init__()

        self.msg = msg

    def __str__(self):
        return self.msg


class NaNError(Exception):
    """Exception when catching NaN value when training"""

    def __init__(self, msg: str):
        super().__init__()

        self.msg = msg

    def __str__(self):
        return self.msg


class AbnormalValueError(Exception):
    """Exception when catching abnormal value when training"""

    def __init__(self, msg: str):
        super().__init__()

        self.msg = msg

    def __str__(self):
        return self.msg