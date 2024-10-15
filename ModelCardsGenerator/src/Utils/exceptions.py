class NoModelException(Exception):
    def __init__(self):
        self.message = "No Model in Model Registry"
        super().__init__(self.message)

class TextValidationError(Exception):
    def __init__(self):
        self.message = "Invalid text format"
        super().__init__(self.message)