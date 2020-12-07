import time
import torch
import torchvision.models as models
from benchmarks.src.models import ReducedVgg16
import torchvision.models as models
from benchmarks.custom_models import vgg16_ext, resnet18_ext
from benchmarks.utils.consts import Split
from benchmarks.src.datasets import ClassificationDataset
from benchmarks.src.input import InputPipeline
from benchmarks.utils.saver import Saver
from benchmarks.utils.tensorboard_writer import SummaryWriter


def _classification_testing(input_pipeline, model, loss_function, optimizer, saver, config_training):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)

    model = saver.load_checkpoint(model, optimizer)[0]

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
    
    # validation epoch
    correct = 0
    total = 0
    losses = 0
    t0 = time.time()
    for idx, (images_batch, labels_batch) in enumerate(input_pipeline[Split.VAL]):
        # Loading tensors in the used device
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

        outputs = model(images_batch)
        loss = loss_function(outputs, labels_batch)

        losses += loss.item()
        local_preds = torch.max(outputs.data, 1)[1]
        total += labels_batch.size(0)
        correct += (local_preds == labels_batch).sum().item()

    val_acc = correct / total
    val_loss = losses / total
    print('VALIDATION :: val Accuracy: {:.4f} - val Loss: {:.4f} in {}s'.format(
        val_acc, val_loss, int(time.time() - t0)
    ))


def classification_test(csv_path, data_folder, model_path, config_training, buckets_files=None):

    batch_size = config_training.getint('batch_size', 256)
    test_ds = ClassificationDataset(Split.TEST, csv_path, data_folder, buckets_files,
        config_training)
    val_ds = ClassificationDataset(Split.VAL, csv_path, data_folder, buckets_files,
        config_training)

    input_pipeline = InputPipeline(datasets_list=[test_ds, val_ds], batch_size=64)

    n_outputs = len(test_ds.get_idx2labels())
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

    saver = Saver(model_path)

    _classification_testing(
        input_pipeline=input_pipeline,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        saver=saver,
        config_training=config_training
    )
