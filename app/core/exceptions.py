class AppError(Exception):
    """Base class for errors we expect and want to turn into clean HTTP responses."""

    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFileError(AppError):
    status_code = 415


class EmptyDocumentError(AppError):
    status_code = 422


class FileTooLargeError(AppError):
    status_code = 413


class LLMError(AppError):
    status_code = 502
