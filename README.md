# Airbus Ship Detection Challenge
Read more information about challenge [here](https://www.kaggle.com/competitions/airbus-ship-detection/overview)

## Model training
This is ML-CV solution for finding ships on satellite images

### Instal requirements

1. Install Python requirements
```bash
pip install -r requirements.txt
```
Specify parameters (path to folders, weight and training parameters) in **preprocessing.py** and **config_unet_model**


### Training

1. Upload data from [here](https://www.kaggle.com/competitions/airbus-ship-detection/data)

2. Read dataset using preprocessing.py

3. Explore the data using explore_visualize.py

4. Display ship segmentation pixels using visualize_ship_segmentation.py

5. Create TensorFlow datasets using config_unet_model.py

6. Set UNet segmentation model and Loss functions using unet_model.py

7. Set class IoU using iou_metric.py

8. Train the model using train_unet.py

9. Plot training and validation loss using predict_visual.py

10. Results analysis submission.py

More information [here](https://t.me/ya_andy_ua)


# Acknowledgements

Special thanks to [Vladyslav_Sh](https://www.kaggle.com/vladyslavsh) for his valuable contributions to the following parts of the code.


# Airbus Ship Detection Challenge
