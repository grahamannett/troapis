import os
import atexit
import functools
from enum import StrEnum, auto

from rich.console import Console
from rich.progress import MofNCompleteColumn, Progress, TimeElapsedColumn
from rich.prompt import Prompt
from rich.markup import escape


# Log levels
class LogLevel(StrEnum):
    """The log levels."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

    def __le__(self, other: "LogLevel") -> bool:
        """Compare log levels.

        Args:
            other: The other log level.

        Returns:
            True if the log level is less than or equal to the other log level.
        """
        levels = list(LogLevel)
        return levels.index(self) <= levels.index(other)


# Set the current log level, allow for environment variable override.
LEVEL = LogLevel(os.environ.get("LOGLEVEL", LogLevel.INFO))

# Console for pretty printing.
_console = Console()


def get_console():
    """helper but not sure if it is useful as can just use _console

    Returns:
        _type_: _description_
    """
    return _console


def setLEVEL(log_level: LogLevel):
    """Set the log level.

    Args:
        log_level: The log level to set.
    """
    global LEVEL
    LEVEL = log_level


def print(msg: str, _stack_offset: int = 2, **kwargs):
    """Print a message.

    Args:
        msg: The message to print.
        kwargs: Keyword arguments to pass to the print function.
    """
    _console.log(msg, _stack_offset=_stack_offset, **kwargs)


def debug(msg: str, **kwargs):
    """Print a debug message.

    Args:
        msg: The debug message.
        kwargs: Keyword arguments to pass to the print function.
    """
    if LEVEL <= LogLevel.DEBUG:
        print(f"[blue]Debug: {msg}[/blue]", **kwargs)


def info(msg: str, _stack_offset: int = 3, **kwargs):
    """Print an info message.

    Args:
        msg: The info message.
        kwargs: Keyword arguments to pass to the print function.
    """
    if LEVEL <= LogLevel.INFO:
        print(f"[cyan]Info: {msg}[/cyan]", _stack_offset=_stack_offset, **kwargs)


def success(msg: str, **kwargs):
    """Print a success message.

    Args:
        msg: The success message.
        kwargs: Keyword arguments to pass to the print function.
    """
    if LEVEL <= LogLevel.INFO:
        print(f"[green]Success: {msg}[/green]", **kwargs)


def log(msg: str, _stack_offset: int = 2, **kwargs):
    """Takes a string and logs it to the console.

    Args:
        msg: The message to log.
        kwargs: Keyword arguments to pass to the print function.
    """
    if LEVEL <= LogLevel.INFO:
        _console.log(msg, _stack_offset=_stack_offset, **kwargs)


def rule(**kwargs):
    """Prints a horizontal rule with a title.

    Args:
        title: The title of the rule.
        kwargs: Keyword arguments to pass to the print function.
    """
    _console.rule(**kwargs)


def warn(msg: str, _stack_offset: int = 3, **kwargs):
    """Print a warning message.

    Args:
        msg: The warning message.
        kwargs: Keyword arguments to pass to the print function.
    """
    if LEVEL <= LogLevel.WARNING:
        print(f"[orange1]Warning: {msg}[/orange1]", _stack_offset=_stack_offset, **kwargs)


@functools.cache
def warning_once(msg: str):
    """Print a warning message once.

    from hf - transformers.
    seems like a useful function to have.

    Args:
        msg: The warning message.
        kwargs: Keyword arguments to pass to the print function.
    """
    warn(msg)


def deprecate(
    feature_name: str,
    reason: str,
    deprecation_version: str,
    removal_version: str,
    **kwargs,
):
    """Print a deprecation warning.

    Args:
        feature_name: The feature to deprecate.
        reason: The reason for deprecation.
        deprecation_version: The version the feature was deprecated
        removal_version: The version the deprecated feature will be removed.
        kwargs: Keyword arguments to pass to the print function.
    """
    msg = (
        f"{feature_name} has been deprecated in version {deprecation_version} {reason.rstrip('.')}. It will be completely "
        f"removed in {removal_version}"
    )
    if LEVEL <= LogLevel.WARNING:
        print(f"[yellow]DeprecationWarning: {msg}[/yellow]", **kwargs)


def error(msg: str, _stack_offset: int = 3, **kwargs):
    """Print an error message.

    Args:
        msg: The error message.
        kwargs: Keyword arguments to pass to the print function.
    """
    if LEVEL <= LogLevel.ERROR:
        print(f"[red]{msg}[/red]", _stack_offset=_stack_offset, **kwargs)


def check_or_fail(condition: bool, msg: str = "Check failed", _stack_offset: int = 3, **kwargs):
    """Check a condition and print an error message if it is False.

    Args:
        condition: The condition to check.
        msg: The error message.
        _stack_offset: The stack offset. 4 is probably the right value.
        kwargs: Keyword arguments to pass to the print function.
    """
    if not condition:
        error(msg, _stack_offset=_stack_offset, **kwargs)
        rule()
        raise AssertionError(msg)


def ask(prompt: str, choices: list[str] = None, default: str = None) -> str:
    """Takes a prompt prompt and optionally a list of choices
     and returns the user input.

    Args:
        prompt: The prompt to ask the user.
        choices: A list of choices to select from.
        default: The default option selected.

    Returns:
        A string with the user input.
    """
    return Prompt.ask(prompt, choices=choices, default=default)  # type: ignore


def _ensure_progress_exit(progress: Progress) -> None:
    """
    Ensure clean exit for progress bar.

    Args:
        progress (Progress): The progress bar to stop.
    """

    def _fn():
        try:
            progress.stop()
        except Exception as err:
            warn(f"Error ensuring progress exits cleanly. Shell cursor may not display. Error: {err}")

    return _fn


def progress(
    ensure_exit: bool = False,
    start: bool = False,
    time_remaining: bool = False,
    **kwargs,
):
    """Create a new progress bar.

    ensure_exit allows for CTRL+C to clean exit and not mess up the terminal cursor

    Default Columns are: TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

    Returns:
        A new progress bar.
    """
    _default_columns = Progress.get_default_columns()

    if not time_remaining:
        _default_columns = _default_columns[:-1]

    pbar = Progress(
        *_default_columns,
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        **kwargs,
    )

    if ensure_exit:
        atexit.register(_ensure_progress_exit(pbar))

    if start:
        pbar.start()

    return pbar


def status(*args, **kwargs):
    """Create a status with a spinner.

    Args:
        *args: Args to pass to the status.
        **kwargs: Kwargs to pass to the status.

    Returns:
        A new status.
    """
    return _console.status(*args, **kwargs)


def with_escape(log_func=info, log_str: str = None, **kwargs):
    """
    Executes the given log function with the provided arguments.
    If no log function is provided, it defaults to the `info` function.

    Args:
        log_func (callable, optional): The log function to execute. Defaults to info.

    Raises:
        TypeError: If the provided log function is not callable.

    Example:
        with_escape(debug, "Debug message")  # Calls the debug function with the message
        with_escape("Error occurred", exc_info=True)  # Calls the info function with the message and exception info
    """
    if not callable(log_func):
        raise TypeError("log_func must be callable")

    log_func(escape(log_str), **kwargs)
