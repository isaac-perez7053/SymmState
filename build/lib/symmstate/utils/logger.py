import logging
from typing import Optional

class Logger:
    _logger: Optional[logging.Logger] = None

    def __init__(self, name: str = "symmstate", level: int = logging.INFO, file_path: Optional[str] = 'symmstate.log'):
        # If no global logger is configured yet, create one.
        if Logger._logger is None:
            Logger._logger = self.configure_logging(name=name, level=level, file_path=file_path)

    @staticmethod
    def configure_logging(name: str, level: int, file_path: Optional[str]) -> logging.Logger:
        """Configure package-wide logging"""
        logger = logging.getLogger(name)
        # Avoid adding duplicate handlers if they already exist.
        if not logger.hasHandlers():
            logger.setLevel(level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Console handler.
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)

            # File handler.
            if file_path:
                fh = logging.FileHandler(file_path)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
        return logger

    @classmethod
    def set_logger(cls, name: str = "symmstate", level: int = logging.INFO, file_path: str = "symmstate.log") -> None:
        """Reconfigure the global logger."""
        cls._logger = cls.configure_logging(name=name, level=level, file_path=file_path)


    @staticmethod
    def log_or_print(
        message: str, 
        logger: logging.Logger = None, 
        level: int = logging.INFO
    ) -> None:
        """
        Logs a message using the global logger by default.
        """
        if logger is None: 
            print(message)
        else: 
            logger = logger 
            logger.log(level, message)

    @property
    def logger(self) -> logging.Logger:
        """Return the global logger."""
        if Logger._logger is None:
            # Fallback: Configure logger if it hasn't been set
            Logger._logger = self.configure_logging(name="symmstate", level=logging.INFO, file_path="symmstate.log")
        return Logger._logger


# For convenience, you can expose a module-level logger instance:
global_logger: logging.Logger = Logger().logger



