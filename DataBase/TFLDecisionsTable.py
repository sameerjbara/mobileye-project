from singletonDecorator import singleton

import pandas as pd

"""
This class represents a DataBase that holds all the information about tfl that founds by the program.
"""


@singleton
class TFLDecisionsTable:
    def __init__(self):
        self.tfl_decision = pd.DataFrame([], columns=["seq", "is_true", "is_ignore", "path",
                                                      "x0", "x1", "y0", "y1", "col"])

    def add_tfls_decisions(self, df: pd.DataFrame):
        self.tfl_decision = pd.concat([self.tfl_decision, df], ignore_index=True)

    def get_tfls_decisions(self):
        return self.tfl_decision

    def print_tfl_decision(self):
        """
        Prints the tfl_decision table.
        :return: None
        """
        print(self.tfl_decision)

    def export_tfls_decisions_to_h5(self):
        df = self.tfl_decision

        df["seq"] = df["seq"].astype(int)
        df["is_true"] = df["is_true"].astype(bool)
        df["is_ignore"] = df["is_ignore"].astype(bool)

        df["x0"] = df["x0"].astype(int)
        df["x1"] = df["x1"].astype(int)

        df["y0"] = df["y0"].astype(int)
        df["y1"] = df["y1"].astype(int)

        df["col"] = df["col"].astype(str)

        df["path"] = df["path"].astype(str)

        df.to_hdf("./Resources/attention_results/crop_results.h5", "crop_results0", format="table")


