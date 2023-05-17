import torch

class DummyScheduler:
    def __init__(self,opt):
        self.lr = opt.param_groups[0]['lr']
        pass
    def step(self):
        pass
    def state_dict(self):
        return None
    def load_state_dict(self, state_dict):
        pass
    def get_last_lr(self):
        return [self.lr]

def get_schedule(opt, sched, EPOCHS=1, epoch_len=0, mile=[.5, .75], gamma=0.1, lr_min=0.0001):
    if sched == 'multisteplr': # 0.1, 0.01, 0.001
        milestones = [round(EPOCHS*epoch_len*mile[i]) for i in range(len(mile))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
    elif sched == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS*epoch_len, eta_min=lr_min)
    else:
        lr_scheduler = DummyScheduler(opt)
    return lr_scheduler