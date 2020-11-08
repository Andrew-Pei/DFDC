import os
import pickle
from glob import glob

import torch
from torch.utils.data import DataLoader
#import torch.nn
import albumentations as A
import sys
sys.path.append("/data1/pbw/DFDC")
from cnn3d.training.i3d import InceptionI3d
from cnn3d.training.datasets_video import RebalancedVideoDataset
from cnn3d.training.utils import Am

"""This is a fairly standard training script which uses:
1) A slightly modified version of the InceptionI3d model from https://github.com/piergiaj/pytorch-i3d
    - The name J3D reflects models where I've made these small changes (sometimes I call it J3D or I3D confusingly).
    - The modifications allow us to feed in videos of various length by ensuring the outputs are either interpolated
      to be 64 frames in length, regardless, or pooled down to a single prediction with 1D globalavgpooling
2) Pretrained weights from the same repository (Apache license) - as ../../external_models/i3d_rgb_charades.pt

For all the 3D models I actually use a categorical crossentropy rather than a binary crossentropy because the I3D model
works by having an output convolutional layer per class.

There sometimes is a little RNG getting this to converge, but it does with some patience.
Interestingly accuracy creeps up slowly before loss start to meaningfully drop.
    ~ 58% train accuracy after 1000 iterations using default settings
    ~ 62% train accuracy after ~ 2000 iterations
I've saved a snapshot of a version in ../data/saved_models as 'j3d_l0.1887_FOR_TL.model' which can be used as an early
snapshot to transfer learn from.
"""

FACE_MP4_DIR = "/data1/pbw/DFDC/cnn3d/data/whole_picture_by_real_fake"  # The directory containing the cropped face MP4s, in 'REAL' and 'FAKE' subdirs
#BATCH_SIZE = 12
BATCH_SIZE = 4
N_WORKERS = 8
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device_ids = [2]
DEVICE = "cuda"
EPOCHS = 20

if __name__ == "__main__":
    train_transforms = A.Compose([
        A.GaussianBlur(p=0.1),
        A.GaussNoise(var_limit=5 / 255, p=0.1),
        A.RandomBrightnessContrast(p=0.25),
        A.RandomResizedCrop(height=224, width=224, scale=(0.5, 1), ratio=(0.9, 1.1)),
        A.HorizontalFlip()])  # We don't normalise in the transforms, we do it in the dataset

    test_transforms = A.Compose([A.CenterCrop(height=224, width=224)])

    test_roots = ['45','46','47','48','49']

    dataset_train = RebalancedVideoDataset(FACE_MP4_DIR, "train", label_per_frame=False, test_videos=test_roots,
                                           transforms=train_transforms, framewise_transforms=True, i3d_norm=True)
    dataset_test = RebalancedVideoDataset(FACE_MP4_DIR, "test", label_per_frame=False, test_videos=test_roots,
                                          transforms=test_transforms, framewise_transforms=True, i3d_norm=True)

    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE*len(device_ids), shuffle=True,
                                  num_workers=N_WORKERS, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE*len(device_ids), shuffle=True, num_workers=N_WORKERS,
                                 pin_memory=True)

    model = InceptionI3d(157, in_channels=3, output_method='avg_pool')
    # model.load_state_dict(torch.load('../../data/external_models/i3d_rgb_charades.pt'))
    model.replace_logits(2)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)
    model.to(DEVICE)
    #model = model.to(DEVICE)
    #model = torch.nn.DataParallel(model, device_ids=device_ids)
    #model = model.cuda()
    #model = model.cuda(device_ids[0])
    #model = torch.nn.DataParallel(model, device_ids=device_ids)
    #model.load_state_dict(torch.load("/data1/pbw/DFDC/cnn3d/saved_models/j3d_e19_l0.1546.model"))

    criterion = torch.nn.CrossEntropyLoss()
    lr = 1e-3

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    #optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    #total_steps=7852,
                                                    max_lr=lr/10,
                                                    steps_per_epoch=len(dataloader_train),
                                                    epochs=EPOCHS)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5)

    for epoch in range(EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {EPOCHS}")
        # TRAIN
        model.train()
        losses, accuracy = Am(), Am()
        for i, (x, y_true) in enumerate(dataloader_train):

            x = x.to(DEVICE)
            #x = x.cuda(device=device_ids[0])
            t = x.size(2)
            y_true = y_true.to(DEVICE)
            #y_true = y_true.cuda(device=device_ids[0])
            y_pred = model(x)

            loss = criterion(y_pred, y_true)
            losses.update(loss.data, x.size(0))

            # Accuracy
            ps = torch.exp(y_pred)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == y_true.view(*top_class.shape)
            accuracy.update(torch.mean(equals.type(torch.FloatTensor)).item())

            optimizer.zero_grad()
            loss.backward()
            #optimizer.module.step()
            optimizer.step()
            
            scheduler.step()

            if i % 10 == 0:
                print(f'TRAIN\t[{i:03d}/{len(dataloader_train):03d}]\t\t'
                      f'LOSS: {losses.running_average:.4f}\t\t'
                      f'ACCU: {accuracy.running_average:.4f}')

        print(f'TRAIN\tComplete!\t\t'
              f'LOSS: {losses.avg:.4f}\t\t'
              f'ACCU: {accuracy.avg:.4f}')

        # TEST
        model.eval()
        losses, accuracy, reals_correct, fakes_correct = Am(), Am(), Am(), Am()
        for i, (x, y_true) in enumerate(dataloader_test):

            with torch.no_grad():

                x = x.to(DEVICE)
                #x = x.cuda(device=device_ids[0])
                t = x.size(2)
                y_true = y_true.to(DEVICE)
                #y_true = y_true.cuda(device=device_ids[0])

                y_pred = model(x)

                loss = criterion(y_pred, y_true)
                losses.update(loss.data, x.size(0))

                # Accuracy
                ps = torch.exp(y_pred)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == y_true.view(*top_class.shape)
                accuracy.update(torch.mean(equals.type(torch.FloatTensor)).item())

                for cls, eq in zip(top_class, equals):  # Iterate over batch
                    if cls[0] == 0:  # Fake
                        fakes_correct.update(bool(eq[0]))
                    elif cls[0] == 1:
                        reals_correct.update(bool(eq[0]))
                    else:
                        print(f"Unknown class!")

                if i % 10 == 0:
                    print(f'TEST\t[{i:03d}/{len(dataloader_test):03d}]\t\t'
                          f'LOSS: {losses.running_average:.4f}\t\t'
                          f'ACCU: {accuracy.running_average:.4f}\t\t'
                          f'REAL: {reals_correct.running_average:.4f}\t'
                          f'FAKE: {fakes_correct.running_average:.4f}')

        print(f'TEST\tComplete!\t\t'
              f'LOSS: {losses.avg:.4f}\t\t'
              f'ACCU: {accuracy.avg:.4f}\t'
              f'REAL: {reals_correct.avg:.4f}\t'
              f'FAKE: {fakes_correct.avg:.4f}')

        torch.save(model.state_dict(), f"/data1/pbw/DFDC/cnn3d/saved_models/j3d_e{epoch}_l{losses.avg:.4f}.model")