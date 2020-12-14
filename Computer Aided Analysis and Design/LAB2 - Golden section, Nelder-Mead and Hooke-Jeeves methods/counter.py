class Function:

    def __init__(self, f):
        self.f = f
        self.counter = 0

    def __call__(self, a):
        self.counter += 1
        return self.f(a)

    def reset(self):
        self.counter = 0