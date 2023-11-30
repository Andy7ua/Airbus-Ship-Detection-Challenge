from train_unet import *
import matplotlib.pyplot as plt

# Plot training and validation loss
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'C2', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.show()

# Plot training and validation mean IoU
mIoU = model_history.history['mean-IoU']
val_mIoU = model_history.history['val_mean-IoU']

plt.figure()
plt.plot(model_history.epoch, mIoU, 'm', label='Training mean IoU')
plt.plot(model_history.epoch, val_mIoU, 'y', label='Validation mean IoU')

plt.title('Training and Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.show()
