import os
from tensorboardX import SummaryWriter as Logger
from benchmarks.utils.consts import Split


class SummaryLogger:
    def __init__(self, logdir, save):
        self.logger = Logger(logdir)
        self.save = save

    def add_scalar(self, name, scalar, global_step):
        if self.save:
            self.logger.add_scalar(name, scalar, global_step)


class SummaryWriter:
    ACC = 'Accuracy'
    ACC_STEPS = 'Accuracy_Steps'
    LOSS = 'Loss'
    LOSS_STEPS = 'Loss_Steps'
    TPA = 'TotalPaddingArea'
    MGPU = 'MemUsedGPU'
    MCPU = 'MemUsedCPU'
    UGPU = 'UtilizationGPU'
    UCPU = 'LoadCPU'
    BATCHSIZE = 'BatchSize'
    DURATION = 'DurationTime'


    def __init__(self, summaries_path, save=False):
        train_logdir = os.path.join(summaries_path, 'train')
        val_logdir = os.path.join(summaries_path, 'val')
        tst_logdir = os.path.join(summaries_path, 'test')
        self.loggers = {
            Split.TRAIN: SummaryLogger(train_logdir, save),
            Split.VAL: SummaryLogger(val_logdir, save),
            #Split.TEST: SummaryLogger(tst_logdir, save)
        }

    def __getitem__(self, subset):
        try:
            return self.loggers[subset]
        except KeyError:
            return None