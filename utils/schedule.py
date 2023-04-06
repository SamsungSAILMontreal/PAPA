import torch

class DummyScheduler:
    def __init__(self):
        pass
    def step(self):
        pass
    def state_dict(self):
        return None
    def load_state_dict(self, state_dict):
        pass


def get_schedule(opt, sched, EPOCHS=1, epoch_len=0, mile=[.5, .75], gamma=0.1):
    if sched == 'multisteplr': # 0.1, 0.01, 0.001
        milestones = [round(EPOCHS*epoch_len*mile[i]) for i in range(len(mile))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
    else:
        lr_scheduler = DummyScheduler()
    return lr_scheduler
