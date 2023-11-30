from config_unet_model import *
import tensorflow_addons as tfa
from iou_metric import IoU
from unet_model import UNetModel
from tensorflow import keras

EPOCHS = 15
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

optimizer = tfa.optimizers.RectifiedAdam(
    learning_rate=0.005,
    total_steps=EPOCHS * STEPS_PER_EPOCH,
    warmup_proportion=0.3,
    min_lr=0.00001,
)
optimizer = tfa.optimizers.Lookahead(optimizer)

loss = tf.keras.losses.CategoricalCrossentropy()
mIoU = IoU(num_classes=2, target_class_ids=[0, 1], sparse_y_true=False, sparse_y_pred=False, name='mean-IoU')

model = UNetModel(IMG_SHAPE + (3,)).model
model.compile(optimizer=optimizer,
              loss=loss, # bce_dice_loss,
              metrics=[mIoU],)

trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
print(f'Trainable params: {trainable_params}')
tf.keras.utils.plot_model(model, show_shapes=True)


checkpoint_filepath = 'model-checkpoint'
save_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_mean-IoU',
    mode='max',
    save_best_only=True
)

model_history = model.fit(train_batches,
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=validation_batches,
                          callbacks=[save_callback])
model.load_weights(checkpoint_filepath)
