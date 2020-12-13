import os

class LogRecorder:
    def __init__(self, path):
        self.path = path
        if  '/' in path:
            pos = path.rfind('/')
            self.dir, self.file = path[:pos], path[pos+1:]
            if not os.path.exists(self.dir):
                os.makedirs(path)
                print("create path: "+self.dir)
        if os.path.exists(path):
            os.remove(path)
            print("remove old file: " + path)

    def write(self, str, addition_std = None, addition_log = None):
        if addition_std is None:
            print(str)
        else:
            print(str+addition_std)
        with open(self.path, "a") as file:
            if addition_log is None:
                file.write(str+"\n")
            else:
                file.write(str+addition_log+"\n")