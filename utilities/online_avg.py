class OnlineAvg:
    def __init__(self):
        self.val, self.n = 0., 0
        
    def __add__(self, other):
        self.n += 1
        self.val = self.val + (other-self.val)/self.n
        return self
    
    def __repr__(self):
        return str(self.val)
    
    def __float__(self):
        return self.val
    
    def __le__(self, other):
        return self.val <= float(other)
    
    def __lt__(self, other):
        return self.val < float(other)