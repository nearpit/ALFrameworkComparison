from utilities import funcs
from core import ActiveLearning


if __name__ == '__main__':
    args = funcs.get_arguments()
    procedure = ActiveLearning(args)
    procedure.run()