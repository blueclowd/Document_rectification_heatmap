import pandas as pd
import os
import random

import keras.backend as K
import pandas as pd
import cv2
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop

from mask_generator import MaskGenerator
from unet import UNET

n_keypoints = 4
image_width = 96
image_height = 96

def jaccard(ytrue, ypred, smooth=1e-5):

    intersection = K.sum(K.abs(ytrue*ypred), axis=-1)
    union = K.sum(K.abs(ytrue)+K.abs(ypred), axis=-1)
    jac = (intersection + smooth) / (union-intersection+smooth)

    return K.mean(jac)


def mean_squared_error(y_true, y_pred):

    channel_loss = K.sum(K.square(y_pred - y_true), axis=-1)
    total_loss = K.mean(channel_loss, axis=-1)
    print("Loss shape:", total_loss.shape)

    return total_loss


def create_callbacks(wts_fn, csv_fn, patience=5, enable_save_wts=True):
    cbks = []

    # early stopping
    early_stopper = EarlyStopping(monitor='val_loss', patience=patience)
    cbks.append(early_stopper)

    # model checkpoint
    if enable_save_wts is True:
        model_chpt = ModelCheckpoint(filepath=wts_fn,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_weights_only=True,
                                     save_best_only=True,
                                     save_freq=patience)

        cbks.append(model_chpt)

    # csv logger
    csv_logger = CSVLogger(csv_fn)
    cbks.append(csv_logger)

    return cbks


def trainModel(model, model_name, loss_type, n_epochs, old_lr, new_lr, load_saved_wts=False):
    if load_saved_wts is True:
        wts_fn = model_name + "_lr=" + str(old_lr) + ".h5"
        model.load_weights(wts_fn)

    wts_fn = model_name + "_lr=" + str(new_lr) + ".h5"
    csv_fn = model_name + "_lr=" + str(new_lr) + ".csv"
    cbks = create_callbacks(wts_fn, csv_fn)

    optim = RMSprop(lr=new_lr)

    model.compile(loss=loss_type, optimizer=optim, metrics=None)
    # model.fit_generator(generator=train_gen,
    #                     validation_data=val_gen,
    #                     epochs=n_epochs,
    #                     callbacks=cbks)

    model.fit(x=train_gen, validation_data=val_gen, epochs=n_epochs, callbacks=cbks)

    return model


data_dir = "data/train"
train_dir = "images"
train_csv = "train.csv"

df_train = pd.read_csv(os.path.join(data_dir, train_csv))

n_train = df_train['file'].size

print('n_train: {}'.format(n_train))

df_kp = df_train.iloc[:, 1:]

idxs = []

img_dict = {}
kp_dict = {}

for i in range(n_train):

    if True in df_train.iloc[i, 1: n_keypoints * 2+1].isna().values:
        continue
    else:
        idxs.append(i)

        img_dict[i] = df_train.iloc[i, 0]

        img = cv2.imread(os.path.join(data_dir, train_dir, img_dict[i]))
        h, w = img.shape[:2]

        # keypoints
        kp = df_kp.iloc[i].values.tolist()
        kp = [a * image_width / w for a in kp]
        kp_dict[i] = kp

random.shuffle(idxs)

cutoff_idx = int(0.9*len(idxs))
train_idxs = idxs[0:cutoff_idx]
val_idxs = idxs[cutoff_idx:len(idxs)]

print("\n# of Training Images: {}".format(len(train_idxs)))
print("# of Val Images: {}".format(len(val_idxs)))

transform_dict = {"Flip": False, "Shift": False, "Scale": False, "Rotate": False}

train_gen = MaskGenerator(os.path.join(data_dir, train_dir),
                              train_idxs,
                              img_dict,
                              kp_dict,
                              transform_dict=transform_dict,
                              augment=False,
                              batch_size=16)

val_gen = MaskGenerator(os.path.join(data_dir, train_dir),
                            val_idxs,
                            img_dict,
                            kp_dict,
                            augment=False,
                            batch_size=16)

print("\n# of training batches= %d" % len(train_gen))
print("# of validation batches= %d" % len(val_gen))

train_imgs, train_masks = train_gen[0]
val_imgs, val_masks = val_gen[0]

loss_type = 'mse'
unet = UNET(input_shape=(image_height, image_width, 1))
print(unet.summary())

unet = trainModel(unet, "unet", loss_type='mean_squared_error', n_epochs=20, old_lr=1e-3, new_lr=1e-3, load_saved_wts=False)