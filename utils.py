class Logger:
    def __init__(self):
        self.log = {}

    def add(self, **kwargs):
        self.log = self.log | kwargs

    def print(self):
        print("----------------")
        for key, val in self.log.items():
            print(f"\033[1m{key}\033[0m: {val}")
        print("----------------")
        print()
