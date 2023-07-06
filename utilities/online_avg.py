class OnlineAvg:
    def __init__(self):
        self.val, self.n = 1.19209e-04, 0 #epsilon
        
    def __add__(self, other):
        self.n += 1
        self.val = self.val + (other-self.val)/self.n
        return self
    
    def __repr__(self):
        return str(self.val)
    
    def __float__(self):
        return self.val