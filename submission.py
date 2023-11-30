from predict_visual import *
from preprocessing import mask_to_rle


def predict(image):
    image = np.expand_dims(image, axis=0)
    pred_mask = model.predict(image)[0].argmax(axis=-1)
    return pred_mask


# Deep Watershed Transform
N = 5

f, ax = plt.subplots(N, 3, figsize=(10, 4 * N))
i = 0
for image, mask in test_dataset.take(N):
    mask = mask.numpy().argmax(axis=-1)
    ax[i, 0].imshow(image)
    ax[i, 0].set_title('image')
    ax[i, 1].imshow(mask)
    ax[i, 1].set_title('true mask')

    pred_mask = predict(image)
    ax[i, 2].imshow(pred_mask)
    ax[i, 2].set_title('predicted mask')
    i += 1

plt.show()

results = model.evaluate(test_batches)
print("test loss, test mIoU:", results)

mIoU = IoU(num_classes=2, target_class_ids=[0, 1], sparse_y_true=True, sparse_y_pred=True, name='mean-IoU')
IoU_results = []
for image, true_mask in test_dataset.take(TEST_LENGTH):
    true_mask = true_mask.numpy().argmax(axis=-1)
    pred_mask = predict(image)
    mIoU.update_state(true_mask, pred_mask)

    iou = IoU(num_classes=2, target_class_ids=[0, 1], sparse_y_true=True, sparse_y_pred=True, name='mean-IoU')
    iou.update_state(true_mask, pred_mask)
    IoU_results.append(iou.result())

plt.hist(IoU_results, bins=15)
print(mIoU.result())

submission = pd.read_csv("sample_submission_v2.csv")


def set_model_prediction(row: pd.Series) -> pd.Series:
    image = cv2.imread(f'{TEST_DIR}{row["ImageId"]}')
    image = cv2.resize(image, IMG_SHAPE, interpolation=cv2.INTER_AREA)
    image = image / 255.0
    pred_mask = predict(image)
    row['EncodedPixels'] = mask_to_rle(pred_mask)
    if row['EncodedPixels'] == '':
        row['EncodedPixels'] = np.nan
    return row


submission = submission.apply(lambda x: set_model_prediction(x), axis=1).set_index("ImageId")

submission.to_csv("./submission.csv")
print(submission)
