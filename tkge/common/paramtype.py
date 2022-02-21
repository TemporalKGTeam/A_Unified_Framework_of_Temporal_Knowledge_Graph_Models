# Base class. Uses a descriptor to set a value
class Descriptor:
    def __init__(self, name: str, default_value=None, **kwargs):
        self.name = name
        self.default_value = default_value
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]


# Descriptor for enforcing types
class Typed(Descriptor):
    expected_type = type(None)

    def __init__(self, name: str, default_value=None, **kwargs):
        if not isinstance(default_value, self.expected_type):
            raise TypeError('Expected ' + str(self.expected_type))

        super().__init__(name=name, default_value=default_value)


    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError('Expected ' + str(self.expected_type))

        super().__set__(instance, value)


# Descriptor for enforcing values
class Unsigned(Descriptor):
    def __set__(self, instance, value):
        if value < 0:
            raise ValueError('Expected >= 0')
        super().__set__(instance, value)


class MaxSized(Descriptor):
    def __init__(self, name=None, **opts):
        if 'size' not in opts:
            raise TypeError('missing size option')
        super().__init__(name, **opts)

    def __set__(self, instance, value):
        if len(value) >= self.size:
            raise ValueError('size must be < ' + str(self.size))
        super().__set__(instance, value)


# Descriptor for enforcing int types
class IntegerParam(Typed):
    expected_type = int


class FloatParam(Typed):
    expected_type = float

class NumberParam(Typed):
    expected_type = (int, float)


class BoolParam(Typed):
    expected_type = bool


class StringParam(Typed):
    expected_type = str


class DeviceParam(Descriptor):
    def __set__(self, instance, value):
        if not isinstance(value, str):
            raise TypeError("Expected a str type")
        if not value.startswith('cpu') and not value.startswith('cuda'):
            raise ValueError("Expected to start with cpu/cuda")
        instance.__dict__[self.name] = value
