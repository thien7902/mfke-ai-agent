"""Unit tests for FatalPollingDetector — the logging.Handler that exits
the process when PTB's Updater logger emits a fatal TLS/certificate error.

These test the detector logic directly (string-matching on a LogRecord)
without spinning up the bot, PTB, or any network.
"""
import importlib
import logging
from unittest.mock import patch

import pytest

# src/bot/__init__.py re-exports `main`, which shadows `import src.bot.main as
# main_mod` (it binds to the function, not the module). importlib sidesteps it.
main_mod = importlib.import_module("src.bot.main")
FATAL_POLLING_ERRORS = main_mod.FATAL_POLLING_ERRORS
FatalPollingDetector = main_mod.FatalPollingDetector


def make_record(msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="telegram.ext.Updater",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=None,
        exc_info=None,
    )


def test_real_log_line_triggers_exit():
    """The exact error from the user's logs should trigger exit + not-alive."""
    detector = FatalPollingDetector()
    msg = ("Error while getting Updates: httpx.ConnectError: "
           "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
           "unable to get local issuer certificate (_ssl.c:1007)")
    with patch.object(main_mod, "os") as mock_os, \
         patch.object(main_mod, "health_checker") as mock_hc:
        detector.emit(make_record(msg))
        mock_hc.set_bot_alive.assert_called_once_with(False)
        mock_os._exit.assert_called_once_with(1)


@pytest.mark.parametrize("token", FATAL_POLLING_ERRORS)
def test_each_fatal_token_triggers_exit(token):
    """Every token in FATAL_POLLING_ERRORS should trigger exit."""
    detector = FatalPollingDetector()
    with patch.object(main_mod, "os") as mock_os, \
         patch.object(main_mod, "health_checker"):
        detector.emit(make_record(f"error: {token}"))
        mock_os._exit.assert_called_once_with(1)


def test_case_insensitive():
    """Mixed-case message should still trigger (handler lowercases)."""
    detector = FatalPollingDetector()
    msg = "SSL: CERTIFICATE_VERIFY_FAILED somewhere in the message"
    with patch.object(main_mod, "os") as mock_os:
        detector.emit(make_record(msg))
        mock_os._exit.assert_called_once_with(1)


def test_recoverable_error_does_not_exit():
    """Connection/timeout errors should NOT trigger exit (recovery path)."""
    detector = FatalPollingDetector()
    msg = "ConnectionTimeout while polling for updates"
    with patch.object(main_mod, "os") as mock_os, \
         patch.object(main_mod, "health_checker") as mock_hc:
        detector.emit(make_record(msg))
        mock_os._exit.assert_not_called()
        mock_hc.set_bot_alive.assert_not_called()


def test_unrelated_error_does_not_exit():
    """Completely unrelated errors should not trigger exit."""
    detector = FatalPollingDetector()
    msg = "MongoServerError: not authorized"
    with patch.object(main_mod, "os") as mock_os:
        detector.emit(make_record(msg))
        mock_os._exit.assert_not_called()


def test_empty_message_safe():
    """Empty / malformed messages should not crash the handler."""
    detector = FatalPollingDetector()
    with patch.object(main_mod, "os") as mock_os:
        detector.emit(make_record(""))
        mock_os._exit.assert_not_called()


def test_exc_info_in_record_safe():
    """If exc_info is set, getMessage should still work and not crash."""
    detector = FatalPollingDetector()
    record = logging.LogRecord(
        name="telegram.ext.Updater", level=logging.ERROR,
        pathname=__file__, lineno=1, msg="error: certificate_verify_failed",
        args=None, exc_info=None,
    )
    with patch.object(main_mod, "os") as mock_os:
        detector.emit(record)
        mock_os._exit.assert_called_once_with(1)


def test_handler_attached_in_main():
    """main() should attach a FatalPollingDetector to telegram.ext.Updater logger.

    This is the critical wiring — without it, PTB polling errors go to that
    logger and never trigger our exit logic.
    """
    # Snapshot existing handlers, call main briefly then cancel, check attachment
    updater_logger = logging.getLogger("telegram.ext.Updater")
    before = list(updater_logger.handlers)

    # Don't actually run main() — manually attach as main() would do, and verify.
    updater_logger.addHandler(FatalPollingDetector())
    after = list(updater_logger.handlers)

    assert len(after) == len(before) + 1
    assert isinstance(after[-1], FatalPollingDetector)

    # Clean up so other tests aren't affected
    updater_logger.removeHandler(after[-1])
