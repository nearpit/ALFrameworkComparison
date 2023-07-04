from acquisitions import Acquisition

# Active Learning via Reconstruction Model
class Alrm(Acquisition):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)