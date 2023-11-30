from preprocessing import *
import matplotlib.pyplot as plt


# Display ship segmentation pixels
def show_image_with_encoded_pixels(image_id: str):
    rows = segmentations[segmentations['ImageId'] == image_id]
    if len(rows) == 0:
        return

    # Read the image corresponding to the image_id
    image = get_train_image(image_id)
    image_size, _, _ = image.shape
    ship_count = len(rows)
    all_ships = np.zeros_like(image)

    # Create subplots for each ship and the combined ships
    ax_rows_number = ship_count + 1
    f, ax = plt.subplots(ax_rows_number, 3, figsize=(15, 5 * ax_rows_number))

    for i in range(ship_count):
        image_info = rows.iloc[i]

        encoded_pixels = np.array(image_info['EncodedPixels'].split(), dtype=int)
        pixels, shift = encoded_pixels[::2], encoded_pixels[1::2]
        ship = np.zeros_like(image)

        for pixel, shift in zip(pixels, shift):
            for j in range(shift):
                cur_pixel = pixel + j - 1
                ship[cur_pixel % image_size, cur_pixel // image_size] = [255, 255, 255]
        all_ships += ship

        # Display the original image, ship mask, and the overlay
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(ship)
        ax[i, 2].imshow(image * (ship // 255))

    # Display the original image, combined ships mask, and the overlay
    ax[ship_count, 0].imshow(image)
    ax[ship_count, 1].imshow(all_ships)
    ax[ship_count, 2].imshow(image * (all_ships // 255))
    plt.show()


# Example usage of the function with different image IDs
image_id = '0006c52e8.jpg'
show_image_with_encoded_pixels(image_id)
print(segmentations[segmentations['ImageId'] == image_id])

image_id = '00113a75c.jpg'
show_image_with_encoded_pixels(image_id)
print(segmentations[segmentations['ImageId'] == image_id])

image_id = '000fd9827.jpg'
show_image_with_encoded_pixels(image_id)
print(segmentations[segmentations['ImageId'] == image_id])

image_id = '007534159.jpg'
show_image_with_encoded_pixels(image_id)
print(segmentations[segmentations['ImageId'] == image_id])
