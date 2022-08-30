from DataBase.TFLCoordinateTable import TFLCoordinateTable as d_tfl
from DataBase.CropImagesTable import CropImagesTable as c_image
import numpy as np
from PIL import Image
import pandas


def get_zoom_rect():
    """
    Gets rows from the database of the tfls coordinates
    """
    for index, row in d_tfl().get_tfls_coordinates().iterrows():
        expended_rect(row, index)


def expended_rect(row, index):
    """
    Get row that represent a rectangle in the image that contain light of traffic light.
    Expend the rectangle to fit to the whole traffic light.
    :param - row: row that represent a rectangle in the image
    :param - index: index of the rectangle in the database
    """
    rect = [[row["y_bottom_left"], row["x_bottom_left"]], [row["y_top_right"], row["x_top_right"]]]
    color = row["col"]

    city = row["path"].split('_')[0]

    image = Image.open("./Resources/leftImg8bit/train/" + city + '/' + row["path"])
    rect_height = rect[0][0] - rect[1][0]
    rect_width = rect[1][1] - rect[0][1]

    if color == "r":
        rect[0][1] -= rect_width * 2
        rect[1][1] += rect_width * 1
        rect[0][0] += rect_width * 5.5
        rect[1][0] -= rect_width * 1.5
    else:
        rect[0][1] -= rect_width * 2
        rect[1][1] += rect_width * 2
        rect_width = rect[1][1] - rect[0][1]
        rect[0][0] += rect_width * 0.5
        rect[1][0] -= rect_width * 2.5

    cropped_image = image.crop((rect[0][1], rect[1][0], rect[1][1], rect[0][0]))
    cropped_image = cropped_image.resize((40, 100))

    cropped_image.save("./Resources/crops/" + row["path"].replace(".png", "_crop_" + str(index) + ".png"))

    zoom = get_zoom_percentage(np.array(image), rect_height * rect_width)
    image_name = row["path"].replace(".png", "_crop_" + str(index) + ".png")

    df = pandas.DataFrame(

        [[index, image_name, round(zoom, 3), rect[0][1], rect[1][1], rect[1][0], rect[0][0], color]],
        columns=["original", "crop_name", "zoom", "x start", "x end", "y start", "y end", "col"])

    c_image().add_crop_image(df)


def get_zoom_percentage(image, rect_area):
    """
    Get the percentage of the rect inside the image
    :param - image: The image to check in.
    :param - rect_area: The rectangle area.
    """
    image_x, image_y, image_z = image.shape
    area_of_image = image_x * image_y
    return rect_area / area_of_image * 100


if __name__ == '__main__':
    get_zoom_rect()
