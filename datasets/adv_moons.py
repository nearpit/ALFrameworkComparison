from datasets.moons import Moons

class Adv_moons(Moons):

    dataset_name = "adv_moons"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    