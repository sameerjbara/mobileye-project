from singletonDecorator import singleton

import pandas as pd

"""
This class represents a DataBase that holds all the information about tfl that founds by the program.
"""


@singleton
class TFLCoordinateTable:
    def __init__(self):
        self.tfl_coordinate = pd.DataFrame([], columns=["path", "y_bottom_left", "x_bottom_left",
                                                        "y_top_right", "x_top_right",
                                                        "col", "RGB", "pixel_light"])

    def get_tfl(self, index):
        return self.tfl_coordinate.iloc[index]

    def get_tfls_coordinates(self):
        """
        :return: DataBase
        """
        return self.tfl_coordinate

    def add_tfl(self, df: pd.DataFrame):
        """
        Adds a given DataFrame to the database.
        :param df: The requested DataFrame to has to the database.
        :return: None
        """
        self.tfl_coordinate = pd.concat([self.tfl_coordinate, df], ignore_index=True)

    def print_tfl_coordinate(self):
        """
        Prints the tfl_coordinate table.
        :return: None
        """
        print(self.tfl_coordinate)

    def export_tfls_coordinates_to_h5(self):
        df = self.tfl_coordinate

        df["x_bottom_left"] = df["x_bottom_left"].astype(int)
        df["y_bottom_left"] = df["y_bottom_left"].astype(int)

        df["x_top_right"] = df["x_top_right"].astype(int)
        df["y_top_right"] = df["y_top_right"].astype(int)
        df["RGB"] = df["RGB"].astype(str)
        df["pixel_light"] = df["pixel_light"].astype(float)
        df["path"] = df["path"].astype(str)
        df["col"] = df["col"].astype(str)

        df.to_hdf("./Resources/attention_results/attention_results.h5", "Traffic_Lights_Coordinates", format="table")
