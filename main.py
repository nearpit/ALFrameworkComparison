import utilities, core 


if __name__ == '__main__':
    args = utilities.get_arguments()
    procedure = core.ActiveLearning(args)
    procedure.run()