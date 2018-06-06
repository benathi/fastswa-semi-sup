class WeightSWA (object):
    """
    SWA or fastSWA
    """
    def __init__(self, swa_model):
        self.num_params = 0
        self.swa_model = swa_model # assume that the parameters are to be discarded at the first update

    def update(self, student_model):
        self.num_params += 1
        print("Updating SWA. Current num_params =", self.num_params)
        if self.num_params == 1:
            print("Loading State Dict")
            self.swa_model.load_state_dict(student_model.state_dict())
        else:
            inv = 1./float(self.num_params)
            for swa_p, src_p in zip(self.swa_model.parameters(), student_model.parameters()):
                swa_p.data.add_(-inv*swa_p.data)
                swa_p.data.add_(inv*src_p.data)
    
    def reset(self):
        self.num_params = 0