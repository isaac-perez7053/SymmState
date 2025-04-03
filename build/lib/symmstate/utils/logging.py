import logging
from typing import Optional

class Logger:
    def __init__(self, name: str, file_path: Optional[str]):
        self.logger = self.configure_logging(name)

    def configure_logging(
        name: str = "symmstate",
        level: int = logging.INFO,
        file_path: Optional[str] = None
    ) -> logging.Logger:
        """Configure package-wide logging"""
        logger = logging.getLogger(name)
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

    # Set new logger variables
    def set_logger(self, name="myapp", level=logging.DEBUG, file_path="app.log"):
        self.logger = self.configure_logging(name=name, level=level, file_path=file_path)