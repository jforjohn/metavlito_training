from time import time
from numpy import prod
import torch
import torchvision.models as models
from benchmarks.custom_models import vgg16_ext, resnet18_ext
from benchmarks.utils.consts import Split
from benchmarks.src.datasets import ClassificationDataset
from benchmarks.src.input import InputPipeline
from benchmarks.utils.saver import Saver
from benchmarks.utils.resources_tracking import get_gpu_used, get_cpu_used
from benchmarks.utils.tensorboard_writer import SummaryWriter

class TrainingLogger:

    def __init__(self, batches_per_epoch, total_prints=9):
        self.batches_per_epoch = batches_per_epoch
        self.total_prints = total_prints
        self.prev_stage = 0

    def get_stage(self, idx):
        return int(idx / self.batches_per_epoch * (self.total_prints + 1))

    def stage_to_string(self, stage):
        return '{:.1f}%'.format(stage / (self.total_prints + 1) * 100)

    def should_print(self, idx):
        stage = self.get_stage(idx)
        if stage == 0:
            return None
        if stage != self.prev_stage:
            self.prev_stage = stage
            return self.stage_to_string(stage)
        return None


def _classification_training(input_pipeline, model, loss_function, optimizer, saver, writer, retrain, config_training):
    print_config = {section: config_training[section] for section in config_training}
    print("Training config '{}'".format(print_config))
    try:
        max_epochs = config_training.getint('max_epochs', 20)
        total_prints = config_training.getint('total_prints', 99)
    except ValueError:
        print("Wrong format in config training.")
        print("Assigning default values")
        max_epochs = 20
        total_prints = 99

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    print('GPUs', torch.cuda.device_count(), flush=True)
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)

    gpucpu_track = config_training.getboolean('gpucpu_track')

    initial_epoch = 0
    step = 0

    if retrain:
        model, optimizer, last_epoch, step = saver.load_checkpoint(model, optimizer)
        initial_epoch = last_epoch + 1

    printer = TrainingLogger(batches_per_epoch=len(input_pipeline[Split.TRAIN]), total_prints=total_prints)

    print('Initial usage:', get_gpu_used(), get_cpu_used())
    start = time()
    for epoch in range(initial_epoch, max_epochs):
        # Training epoch
        correct = 0
        total = 0
        losses = 0
        t0 = time()
        
        for idx, (step_images, step_labels) in enumerate(input_pipeline[Split.TRAIN]):
            #print((epoch, idx, 'training', mem), flush=True)
            # Loading tensors in the used device
            gpu_mem_used, gpu_util = get_gpu_used()
            cpu_mem_used, cpu_util = get_cpu_used()
            step_batchsize = prod(step_images[0].size())

            step_images, step_labels = step_images.to(device), step_labels.to(device)
            #del batch_images
            #print(flush=True)
            #print((epoch, idx, step_images.size()), flush=True)
            #print(flush=True)
            # zero the parameter gradients
            optimizer.zero_grad()
            step_output = model(step_images)
            loss = loss_function(step_output, step_labels)

            loss.backward()
            optimizer.step()

            step += 1

            step_preds = torch.max(step_output.data, 1)[1]
            step_correct = (step_preds == step_labels).sum().item()
            step_time = time() - start

            if gpucpu_track:
                step_mgpu, step_ugpu = get_gpu_used()
                step_mcpu, step_ucpu = get_cpu_used()
                
            step_total = step_labels.size(0)
            step_loss = loss.item()
            step_acc = step_correct / step_total

            losses += step_loss
            total += step_total
            correct += step_correct
            torch.cuda.empty_cache()
            print_stage = printer.should_print(idx)
            if print_stage:
                print('Epoch, Step: ({},{}) => ({}) Train Accuracy: {:.4f} - Train Loss: {:.4f} at {}s - Usage GPU:(before->{:.2f}MB {:.2f}%, after->{:.2f}MB {:.2f}%), CPU:(before->{:.2f}MB {:.2f}%, after->{:.2f}MB {:.2f}%)'.format(
                    epoch, step-1, print_stage, step_acc, step_loss, int(time() - t0), gpu_mem_used, gpu_util, step_mgpu, step_ugpu, cpu_mem_used, cpu_util, step_mcpu, step_ucpu), flush=True
                )
            writer[Split.TRAIN].add_scalar(writer.ACC_STEPS, step_acc, global_step=step)
            writer[Split.TRAIN].add_scalar(writer.LOSS_STEPS, step_loss, global_step=step)
            writer[Split.TRAIN].add_scalar(writer.BATCHSIZE, step_batchsize, global_step=step)
            writer[Split.TRAIN].add_scalar(writer.DURATION, step_time, global_step=step)
            if gpucpu_track:
                writer[Split.TRAIN].add_scalar(writer.MGPU, step_mgpu, global_step=step)
                writer[Split.TRAIN].add_scalar(writer.UGPU, (step_ugpu+gpu_util)/2, global_step=step)
                writer[Split.TRAIN].add_scalar(writer.MCPU, step_mcpu, global_step=step)
                writer[Split.TRAIN].add_scalar(writer.UCPU, step_ucpu, global_step=step)

        train_acc = correct / total
        train_loss = losses / total
        print('EPOCH {} :: Train Accuracy: {:.4f} - Train Loss: {:.4f} in {}s'.format(
            epoch, train_acc, train_loss, int(time() - t0)
        ))
        writer[Split.TRAIN].add_scalar(writer.ACC, train_acc, global_step=step)
        writer[Split.TRAIN].add_scalar(writer.LOSS, train_loss, global_step=step)

        saver.save_checkpoint(model, optimizer, epoch, step)

        # Validation epoch
        correct = 0
        total = 0
        losses = 0
        t0 = time()
        with torch.no_grad():
            for step_images, step_labels in input_pipeline[Split.VAL]:
                step_batchsize = prod(step_images[0].size())
                gpu_mem_used, gpu_util = get_gpu_used()

                # Loading tensors in the used device
                step_images, step_labels = step_images.to(device), step_labels.to(device)

                step_output = model(step_images)
                loss = loss_function(step_output, step_labels)

                step_preds = torch.max(step_output.data, 1)[1]
                step_correct = (step_preds == step_labels).sum().item()
                step_total = step_labels.size(0)
                step_loss = loss.item()
                step_time = time() - t0
                if gpucpu_track:
                    step_mgpu, step_ugpu = get_gpu_used()
                    step_mcpu, step_ucpu = get_cpu_used()

                losses += step_loss
                total += step_total
                correct += step_correct

        val_acc = correct / total
        val_loss = losses / total
        print('EPOCH {} :: Validation Accuracy: {:.4f} - Validation Loss: {:.4f} in {}s'.format(
            epoch, val_acc, val_loss, int(time() - t0)
        ))
        writer[Split.VAL].add_scalar(writer.ACC, val_acc, global_step=step)
        writer[Split.VAL].add_scalar(writer.LOSS, val_loss, global_step=step)
        writer[Split.VAL].add_scalar(writer.BATCHSIZE, step_batchsize, global_step=step)
        writer[Split.VAL].add_scalar(writer.DURATION, step_time, global_step=step)
        if gpucpu_track:
                writer[Split.VAL].add_scalar(writer.MGPU, step_mgpu, global_step=step)
                writer[Split.VAL].add_scalar(writer.UGPU, (step_ugpu+gpu_util)/2, global_step=step)
                writer[Split.VAL].add_scalar(writer.MCPU, step_mcpu, global_step=step)
                writer[Split.VAL].add_scalar(writer.UCPU, step_ucpu, global_step=step)
    
    '''
    # testing epoch
    correct = 0
    total = 0
    losses = 0
    t0 = time.time()
    with torch.no_grad():
        # testing epoch
        correct = 0
        total = 0
        losses = 0
        t0 = time.time()
        for idx, (images_batch, labels_batch) in enumerate(input_pipeline[Split.TEST]):
            # Loading tensors in the used device
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

            outputs = model(images_batch)
            loss = loss_function(outputs, labels_batch)

            losses += loss.item()
            local_preds = torch.max(outputs.data, 1)[1]
            total += labels_batch.size(0)
            correct += (local_preds == labels_batch).sum().item()

        test_acc = correct / total
        test_loss = losses / total
        print('TESTING :: test Accuracy: {:.4f} - test Loss: {:.4f} in {}s'.format(
            test_acc, test_loss, int(time.time() - t0)
        ))
    '''

def classification_train(csv_path, data_folder,
                        model_path, summaries_path,
                        config_training,
                        retrain=False, buckets_files=None):
    batch_size = config_training.getint('batch_size', 256)    

    train_ds = ClassificationDataset(
        Split.TRAIN, csv_path,
        data_folder, buckets_files,
        config_training)
    val_ds = ClassificationDataset(
        Split.VAL, csv_path,
        data_folder, buckets_files,
        config_training)
    #test_ds = ClassificationDataset(
    #    Split.TEST, csv_path,
    #    data_folder, buckets_files,
    #    config_training)

    save_files = config_training.getboolean('save_checkpoints')
    # Model Saver
    saver = Saver(model_path, save_files)

    writer = SummaryWriter(summaries_path, save_files)

    gpu_track = config_training.getboolean('gpucpu_track')
    input_writer = writer if gpu_track else None
    input_pipeline = InputPipeline(
        datasets_list=[train_ds, val_ds],
        batch_size=batch_size,
        writer=input_writer
        )

    n_outputs = len(train_ds.get_idx2labels())
    model_name = config_training.get('model')
    if model_name == 'vgg16':
        model = vgg16_ext(num_classes=n_outputs)
    elif model_name == 'resnet18':
        model = resnet18_ext(num_classes=n_outputs)
    elif model_name == 'densenet121':
        model = models.densenet121(num_classes=n_outputs)
    elif model_name == 'alexnet':
        model = models.alexnet(num_classes=n_outputs)

    learning_rate = config_training.getfloat('learning_rate', 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_function = torch.nn.CrossEntropyLoss(reduction='sum')


    torch.backends.cudnn.benchmark = True

    _classification_training(
        input_pipeline=input_pipeline,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        saver=saver,
        writer=writer, 
        retrain=retrain,
        config_training=config_training
    )
