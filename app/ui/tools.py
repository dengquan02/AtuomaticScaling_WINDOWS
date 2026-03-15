"""Light-weight enums/constants for GUI tools."""

from enum import Enum


class ToolMode(str, Enum):
    SELECT = "select"
    ADD = "add"


class Layer(str, Enum):
    E = "E"
    F1 = "F1"
    F2 = "F2"

    @classmethod
    def list(cls):
        return [cls.E.value, cls.F1.value, cls.F2.value]
