import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure

from DataBase.CropImagesTable import CropImagesTable as c_image
from DataBase.TFLDecisionsTable import TFLDecisionsTable as d_tfl
from DataBase.TFLCoordinateTable import TFLCoordinateTable as t_tfl

def crops_validation():
    # Get Data of crop images


    decisions = []

    for index, row in c_image().get_crops_images().iterrows():
        image_name = c_image().get_crops(row["original"])["crop_name"]

        crop = c_image().get_crops(row["original"])
        x0 = crop["x start"]
        x1 = crop["x end"]
        y0 = crop["y start"]
        y1 = crop["x end"]
        col = crop["col"][0].lower()

        result = is_valid(row)
        if result:
            decisions.append([index, True, False, image_name, x0, x1, y0, y1, col])
        elif result == None:
            decisions.append([index, True, True, image_name, x0, x1, y0, y1, col])
        elif result == False:
            decisions.append([index, False, False, image_name, x0, x1, y0, y1, col])

    d_tfl().add_tfls_decisions(pd.DataFrame(decisions, columns=["seq", "is_true", "is_ignore", "path",
                                                           "x0", "x1", "y0", "y1", "col"]))


def is_valid(crop_data: pd.Series):
    # Need to build a function for the first DataBase to find original image by given index.

    image_name = t_tfl().get_tfl(crop_data["original"])["path"]

    city = image_name.split('_')[0]

    color_image_path = image_name.replace("leftImg8bit.png", "gtFine_color.png")
    label_image_path = image_name.replace("leftImg8bit.png", "gtFine_labelIds.png")

    color_image = np.array(Image.open("./Resources/gtFine/train/" + city + '/' + color_image_path))
    label_image = np.array(Image.open("./Resources/gtFine/train/" + city + '/' + label_image_path).convert("L"))

    color_image_crop = color_image[int(crop_data["y start"]):int(crop_data["y end"]),
                       int(crop_data["x start"]):int(crop_data["x end"])]

    blobs_labels = measure.label(label_image, background=0)

    first_tfl_pixel = None
    valid_pixels = 0

    for row in range(int(crop_data["y start"]), int(crop_data["y end"])):
        if row < 0:
            continue
        elif row >= color_image.shape[0]:
            break
        for col in range(int(crop_data["x start"]), int(crop_data["x end"])):
            if 0 < col < color_image.shape[1]:
                if np.array_equal(color_image[row][col], np.array([250, 170, 30, 255])):
                    if not first_tfl_pixel:
                        first_tfl_pixel = (row, col)
                    valid_pixels += 1

    if not first_tfl_pixel:
        return False

    component_num = blobs_labels[first_tfl_pixel[0]][first_tfl_pixel[1]]
    component_index = np.argwhere(blobs_labels == component_num)
    component_size = component_index.shape[0]

    result = (valid_pixels / component_size) * 100
    area_cropped_image = color_image_crop.shape[0] * color_image_crop.shape[1]
    if area_cropped_image == 0:
        return False
    # result_2 = (valid_pixels / area_cropped_image) * 100

    if result >= 70:
        return True
    elif 15 < result < 70:
        return None
    else:
        return False


if __name__ == '__main__':
    crops_validation()
