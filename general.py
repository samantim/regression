import time

class Logger:
    logged : str 
    def __init__(self):
        self.logged = ""
        self.current_time = time.time()

    def log(self, msg : str):
        self.logged += str(msg) + "\n"

    def save_file(self, file_path : str):
        with open(file_path, "w") as file:
            file.write(self.logged)

    def print_elapsed_time(self, suffix : str):
        elapsedtime = int(time.time() - self.current_time)
        self.current_time = time.time()
        print(f"Elapsed time for {suffix}: {elapsedtime} seconds")