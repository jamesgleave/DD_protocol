class Error(Exception):
    """Base class for other exceptions"""
    pass


class IncorrectModelModeError(Error):
    """Exception raised for errors in the model mode.

    Attributes:
        mode -- input mode which caused the error
        message -- explanation of the error
    """
    def __init__(self, mode, available_modes, message="Incorrect model mode. Use one of the following modes:"):
        self.mode = mode
        self.message = message
        self.available_modes = available_modes

    def __str__(self):
        mode_string = "\n\n"
        for mode in self.available_modes:
            mode_string += " " + mode + "\n"

        return f'{self.mode} -> {self.message}' + mode_string
