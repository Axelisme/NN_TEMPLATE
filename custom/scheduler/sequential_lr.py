
from typing import List, Dict

from torch.optim.lr_scheduler import SequentialLR

from util.utility import create_instance

class MySequentialLR(SequentialLR):
    def __init__(self, optimizer, scheduler_confs: List[Dict], *args, **kwargs):

        scheds = []
        for scheduler_conf in scheduler_confs:
            scheds.append(create_instance(scheduler_conf, optimizer=optimizer))

        super(MySequentialLR, self).__init__(
            optimizer=optimizer,
            schedulers=scheds,
            *args, **kwargs
        )