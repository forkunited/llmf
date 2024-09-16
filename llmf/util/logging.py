"""
Utilities for logging
"""

from attrs import define
from enum import Enum

from datetime import datetime
import json

from typing import Iterator, List, Optional, Set, Tuple, Union

_NEWLINE_SPLIT = "\n"
_MULTILINE_OUTPUT_PREFIX = "\n>>> "

@define
class _LogConfig:
    """ Log configuration """
    debug: bool = False
    log_file_path: Optional[str] = None

class LogEntryType(Enum):
    """ Type of log entry """
    MESSAGE = "MESSAGE"
    OBJECT = "OBJECT"

@define(frozen=True)
class LogEntry:
    """ Log entry """
    time: datetime
    entry_type: LogEntryType
    source: str
    key: str
    message_or_object: Union[str, dict]

    def pretty(self, show_character_counts: bool = False) -> str:
        """ Pretty string representation of the log """
        entry_parts = [
            f"Time: {str(self.time)}",
            f"Source: {self.source}",
            f"Key: {self.key}",
        ]

        if self.entry_type == LogEntryType.MESSAGE:
            character_count_str = "" if not show_character_counts else \
                f" ({len(self.message_or_object)})"
            entry_parts.append(f"Message{character_count_str}: {self.message_or_object}")
        else:
            for obj_key in sorted(self.message_or_object.keys()):
                obj_value = str(self.message_or_object[obj_key])
                character_count_str = "" \
                    if not show_character_counts or not isinstance(self.message_or_object[obj_key], str) else \
                    f" ({len(obj_value)})"
                if "\n" in obj_value:
                    entry_parts.append(
                        f"{obj_key}{character_count_str}:\n{_MULTILINE_OUTPUT_PREFIX}{_MULTILINE_OUTPUT_PREFIX.join(obj_value.split(_NEWLINE_SPLIT))}"
                    )
                else:
                    entry_parts.append(f"{obj_key}: {obj_value}")
        return "\n".join(entry_parts)



    def __str__(self) -> str:
        """ String representation of the entry """
        return "\t".join((
            str(self.time),
            self.entry_type.value,
            self.source,
            self.key,
            (
                self.message_or_object
                if isinstance(self.message_or_object, str)
                else json.dumps(self.message_or_object)
            )
        ))

    @classmethod
    def from_string(cls, entry_str: str) -> "LogEntry":
        """ Construct an entry from a string """
        entry_parts = entry_str.strip().split("\t")
        entry_type = LogEntryType(entry_parts[1].strip())
        return LogEntry(
            time=datetime.fromisoformat(entry_parts[0].strip()),
            entry_type=entry_type,
            source=entry_parts[2].strip(),
            key=entry_parts[3].strip(),
            message_or_object=(
                json.loads(entry_parts[4].strip())
                if entry_type == LogEntryType.OBJECT else
                entry_parts[4].strip()
            )
        )

_CONFIG = _LogConfig()

def config(debug: bool = False, log_file_path: Optional[str] = None):
    """ Configure the logger"""
    _CONFIG.debug = debug
    _CONFIG.log_file_path = log_file_path

def info(source: str, key: str, message_or_object: Union[str, dict]):
    """ Log a message or JSON object """

    log_entry = LogEntry(
        time=datetime.now(),
        entry_type=(
            LogEntryType.MESSAGE if isinstance(message_or_object, str)
            else LogEntryType.OBJECT
        ),
        source=source,
        key=key,
        message_or_object=message_or_object
    )

    if _CONFIG.debug:
        print(f"{log_entry.pretty()}\n")
    if _CONFIG.log_file_path is not None:
        with open(_CONFIG.log_file_path, "a") as fp:
            fp.write(f"{str(log_entry)}\n")

def collect_sources_and_keys() -> List[Tuple[str, str]]:
    """ Show source and key names from the logs """
    sources_and_keys: Set[Tuple[str, str]] = set()
    with open(_CONFIG.log_file_path, "r") as fp:
        for log_line in fp:
            log_entry = LogEntry.from_string(log_line)
            sources_and_keys.add((log_entry.source, log_entry.key))
    return sorted(list(sources_and_keys))

def select(
    source: Optional[str] = None,
    key: Optional[str] = None,
    limit: Optional[int] = None,
    descending: bool = True
) -> Iterator[LogEntry]:
    """ Select entries from the log """
    if _CONFIG.log_file_path is None:
        return
    if limit is not None and limit <= 0:
        return

    # Assume the log is small for now.  Eventually read
    # in memory efficient way if not small later
    with open(_CONFIG.log_file_path, "r") as fp:
        count: int = 0
        for log_line in (reversed(list(fp)) if descending else fp):
            log_entry = LogEntry.from_string(log_line)

            if source is not None and log_entry.source != source:
                continue
            if key is not None and log_entry.key != key:
                continue

            yield log_entry

            count += 1
            if limit is not None and count >= limit:
                return
