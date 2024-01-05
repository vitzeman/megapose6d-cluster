import os
import json

import numpy as np

HORIZONTAL_LINE = "\\hline"
END_ROW = "\\\\"


def create_latex_notation_overall(path2folders, name_json="scores_bop19.json"):
    contains = sorted(os.listdir(path2folders))
    annotation_line = (
        "method & AR & $\\mathrm{AR_{VSD}}$ & $\\mathrm{AR_{MSPD}}$ & $\\mathrm{AR_{MSSD}}$ "
        + END_ROW
        + HORIZONTAL_LINE
    )
    latex_table = [annotation_line]
    results = {}
    for folder in contains:
        method = folder.split("_")[0]
        path2json = os.path.join(path2folders, folder, name_json)
        with open(path2json) as f:
            scores = json.load(f)
        results[method] = scores

    for method, scores in results.items():
        AR = scores["bop19_average_recall"] * 100
        AR_VSD = scores["bop19_average_recall_vsd"] * 100
        AR_MSPD = scores["bop19_average_recall_mspd"] * 100
        AR_MSSD = scores["bop19_average_recall_mssd"] * 100
        line = f"{method} & {AR:.2f} & {AR_VSD:.2f} & {AR_MSPD:.2f} & {AR_MSSD:.2f} " + END_ROW
        latex_table.append(line)

    latex_table = "\n".join(latex_table)
    print(latex_table)


def create_latex_notation_per_object(path2result):
    for method_folder in sorted(os.listdir(path2result)):
        method = method_folder.split("_")[0]
        contains = sorted(os.listdir(os.path.join(path2result, method_folder)))
        for error_metric_folder in contains:
            error_metric = error_metric_folder.split("_")[0].split("=")[1]
            if error_metric in ["mspd", "mssd"]:
                inside = sorted(
                    os.listdir(os.path.join(path2result, method_folder, error_metric_folder))
                )
                for threshold_file in inside:
                    name_threshold = threshold_file.split("_")[0]
                    if name_threshold != "score":
                        continue
                    path2json = os.path.join(
                        path2result, method_folder, error_metric_folder, threshold_file
                    )


if __name__ == "__main__":
    path2folders = os.path.join("Tags", "eval")

    # create_latex_notation_overall(path2folders)

    # create_latex_notation_per_object(path2folders)
    dict = {
        "obj_recalls": {
            "1": 0.1323529411764706,
            "2": 0.008238928939237899,
            "3": 0.004198740377886634,
            "4": 0.0,
            "5": 0.0,
            "6": 0.00429553264604811,
            "7": 0.0,
            "8": 0.0,
        },
        "recall": 0.017586474931986007,
    }
    recall = dict["recall"]
    obj_recalls = dict["obj_recalls"]
    obj_recalls = [obj_recall for _, obj_recall in obj_recalls.items()]

    print(recall)
    print(np.mean(obj_recalls))
