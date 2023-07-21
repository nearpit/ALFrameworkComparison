class OnlineAvg:
    def __init__(self):
        self.val, self.n = 0., 1
        
    def __add__(self, other):
        self.val = self.val + (other-self.val)/self.n
        self.n += 1
        return self
    
    def __repr__(self):
        return str(self.val)
    
    def __float__(self):
        return self.val
    
    def __int__(self):
        return self.n
    
    def __sub__(self, other):
        return float(self.val - float(other))
    
    def __le__(self, other):
        return self.val <= float(other)
    
    def __lt__(self, other):
        return self.val < float(other)