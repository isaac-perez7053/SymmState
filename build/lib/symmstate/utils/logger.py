import logging
from typing import Optional

class Logger:
    _logger: Optional[logging.Logger] = None

    def __init__(self, name: str = "symmstate", level: int = logging.INFO, file_path='symmstate.log'):
        if Logger._logger is None:
            Logger._logger = self.configure_logging(name=name, level=level, file_path=file_path)

    @staticmethod
    def configure_logging(
        name: str,
        
        level: int,
        file_path: Optional[str]
    ) -> logging.Logger:
        """Configure package-wide logging"""
        logger = logging.getLogger(name)
        
        # Check if the logger already has handlers configured to avoid duplicates
        if not logger.handlers:
            logger.setLevel(level)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)

            # File handler
            if file_path:
                fh = logging.FileHandler(file_path)
                fh.setFormatter(formatter)
                logger.addHandler(fh)

        return logger

    @classmethod
    def set_logger(cls, name="symmstate", level=logging.INFO, file_path="symmstate.log"):
        cls._logger = cls.configure_logging(name=name, level=level, file_path=file_path)

    @property
    def logger(self):
        return self._logger


