from singletonDecorator import singleton

import pandas as pd

"""
This class represents a DataBase that holds all the information about tfl that founds by the program.
"""


@singleton
class CropImagesTable:
    def __init__(self):
        self.crop_images = pd.DataFrame([]
                                        , columns=["original", "crop_name", "zoom",
                                                   "x start", "x end", "y start", "y end", "col"])

    def add_crop_image(self, df: pd.DataFrame):
        self.crop_images = pd.concat([self.crop_images, df], ignore_index=True)

    def get_crops_images(self):
        return self.crop_images

    def get_crops(self, index):
        return self.crop_images.iloc[index]

    def print_crop_images(self):
        """
        Prints the crop_images table.
        :return: None
        """
        print(self.crop_images)

    def export_crops_images_to_h5(self):
        df = self.crop_images

        df["x start"] = df["x start"].astype(int)
        df["x end"] = df["x end"].astype(int)

        df["y start"] = df["y start"].astype(int)
        df["y end"] = df["y end"].astype(int)
        df["col"] = df["col"].astype(str)
        df["zoom"] = df["zoom"].astype(float)
        df["original"] = df["original"].astype(str)
        df["crop_name"] = df["crop_name"].astype(str)

        df.to_hdf("./Resources/attention_results/crop_results0.h5", "Traffic_Lights_Coordinates", format="table")
