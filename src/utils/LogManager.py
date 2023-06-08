

class LogManager:
    def __init__(self, file_name, file_path):
        self.file_name = file_name
        self.file_path = file_path
        self.logs_line = []

    def print_(self, logs, print_out=True):
        self.logs_line.append(logs)
        if print_out:
            print(logs)

    def save_log(self):
        pass
