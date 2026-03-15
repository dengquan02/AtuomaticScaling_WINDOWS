"""Undo/redo stack for interactive editing."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .state import Document


class Command:
    """Base command with execute/undo hooks."""

    description: str = ""

    def execute(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def undo(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class AddPointCommand(Command):
    doc: Document
    layer: str
    point: Tuple[float, float]
    index: Optional[int] = None
    _final_index: Optional[int] = None
    description: str = "Add point"

    def execute(self) -> None:
        self._final_index = self.doc.insert_point(self.layer, self.point, self.index)

    def undo(self) -> None:
        if self._final_index is not None:
            self.doc.delete_point(self.layer, self._final_index)


@dataclass
class MovePointCommand(Command):
    doc: Document
    layer: str
    index: int
    new_point: Tuple[float, float]
    old_point: Tuple[float, float]
    description: str = "Move point"

    def execute(self) -> None:
        self.doc.move_point(self.layer, self.index, self.new_point)

    def undo(self) -> None:
        self.doc.move_point(self.layer, self.index, self.old_point)


@dataclass
class DeletePointCommand(Command):
    doc: Document
    layer: str
    index: int
    removed_point: Optional[Tuple[float, float]] = None
    description: str = "Delete point"

    def execute(self) -> None:
        self.removed_point = self.doc.delete_point(self.layer, self.index)

    def undo(self) -> None:
        if self.removed_point is not None:
            self.doc.insert_point(self.layer, self.removed_point, self.index)


class HistoryManager:
    def __init__(self) -> None:
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []

    def run(self, command: Command) -> None:
        command.execute()
        self._undo_stack.append(command)
        self._redo_stack.clear()

    def undo(self) -> Optional[str]:
        if not self._undo_stack:
            return None
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        return cmd.description

    def redo(self) -> Optional[str]:
        if not self._redo_stack:
            return None
        cmd = self._redo_stack.pop()
        cmd.execute()
        self._undo_stack.append(cmd)
        return cmd.description

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
