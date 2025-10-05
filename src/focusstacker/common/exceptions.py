class FocusStackerException(Exception):
    """Base exception for focus stacker."""

    pass


class FocusStackerValidationException(FocusStackerException):
    """Exception raised when a validation error occurs."""

    pass


class FocusStackerAlignmentException(FocusStackerException):
    """Exception raised when image alignment fails."""

    pass


class FocusStackerStackingException(FocusStackerException):
    """Exception raised when image stacking fails."""

    pass


class FocusStackerFileException(FocusStackerException):
    """Exception raised when file operations fail."""

    pass


class FocusStackerMemoryException(FocusStackerException):
    """Exception raised when memory or resource limits are exceeded."""

    pass


class FocusStackerConfigurationException(FocusStackerException):
    """Exception raised when configuration parameters are invalid."""

    pass


class FocusStackerImageProcessingException(FocusStackerException):
    """Exception raised when image processing operations fail."""

    pass


class FocusStackerDirectoryException(FocusStackerException):
    """Exception raised when directory operations fail."""

    pass
