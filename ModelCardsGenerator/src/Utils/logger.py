class Logger:
    def __init__(self):
        self.messages = []
    
    def log(self, msg):
        self.messages.append(msg)
    
    def display(self):
        out = None
        error = False
        if self.messages:
            for msg in self.messages:
                if msg.startswith(("Check", "Exception")):
                    out = "Error:\n"
                    out += f"- {msg}\n"
                    error = True
                    print(out)

            if not error:
                out = "Warnings:\n"
                for i, msg in enumerate(self.messages):
                    out += f"{i + 1}. {msg}\n"
                print(out)
