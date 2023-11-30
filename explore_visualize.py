from preprocessing import *
import matplotlib.pyplot as plt

# Corrupted images
corrupted_images = ['6384c3e78.jpg']  # Premature end of JPEG file error occurs when reading the file
print(segmentations[segmentations['ImageId'].isin(corrupted_images)])

# Delete the corrupted image
segmentations = segmentations.drop(segmentations[segmentations['ImageId'].isin(corrupted_images)].index)
print(segmentations[segmentations['ImageId'].isin(corrupted_images)])


# Exploring the data
print(f'There are {segmentations.shape[0]} rows.')
print(segmentations.head(10))

train_images_number = segmentations['ImageId'].nunique()
print(f'There are {train_images_number} train images.')

# Image resolution
print(segmentations['ImageHeight'].value_counts())
print(segmentations['ImageWidth'].value_counts())


# Distribution of the number of ships in images
images_without_ships = segmentations['EncodedPixels'].isna().sum()
print(f'There are {images_without_ships} images without ships.')

# Create a new column 'ShipCount' to represent the number of ships in each image
segmentations['ShipCount'] = segmentations.apply(lambda x: 0 if pd.isna(x['EncodedPixels']) else 1, axis=1)
# Calculate and visualize the distribution of the number of ships
ships_numbers = segmentations[['ImageId','ShipCount']].groupby(['ImageId']).sum()
print(ships_numbers.value_counts())

# Plot histogram and pie chart for ship counts
f, ax = plt.subplots(1, 2,figsize=(20,10))

ships_numbers.hist(bins = 15, ax=ax[0])

y = ships_numbers.value_counts().values
percent = 100.*y/y.sum()
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(ships_numbers.value_counts().index.get_level_values(0), percent)]
ships_numbers.value_counts().plot.pie(labels=None, ax=ax[1])
ax[1].legend(labels, bbox_to_anchor=(1., 1.), fontsize=14)
ax[1].yaxis.set_visible(False)
ax[1].set_title('Distribution number of ships')

plt.show()

# Plot histograms and box plots for ShipAreaPercentage
f, ax = plt.subplots(1, 3,figsize=(30,10))

segmentations['ShipAreaPercentage'].hist(bins=20, ax=ax[0])
segmentations['ShipAreaPercentage'].plot.box(ax=ax[1])
ax[1].set_ylabel('Ship Area Percentage')
ax[1].set_xlabel('')
segmentations['ShipAreaPercentage'].apply(lambda x: x ** 0.5).plot.box(ax=ax[2])
ax[2].set_ylabel('Sqrt(Ship Area Percentage)')
ax[2].set_xlabel('')

plt.show()
