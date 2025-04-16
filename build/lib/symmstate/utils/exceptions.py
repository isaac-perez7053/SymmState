class SymmStateError(Exception):
    """Base exception for the package"""


class ConvergenceError(SymmStateError):
    """Raised when convergence criteria not met"""


class ParsingError(SymmStateError):
    """Raised when file parsing fails"""


class JobSubmissionError(SymmStateError):
    """Raised when job submission fails"""
