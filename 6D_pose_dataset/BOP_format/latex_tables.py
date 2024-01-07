import os
import json

import numpy as np

HORIZONTAL_LINE = "\\hline"
END_ROW = "\\\\"

LABELS = {
    1: "d01\_controller",
    2: "d02\_servo",
    3: "d03\_main",
    4: "d04\_motor",
    5: "d05\_axle\_front",
    6: "d06\_battery",
    7: "d07\_axle\_rear",
    8: "d08\_chassis",
}
TABLE_HEADER_LINE_1 = f"\\begin{{table}}[ht]"
TABLE_HEADER_LINE_2 = f"\\centering"
TABLE_HEADER_LINE_3 = f"\\begin{{tabular}}{{cc|cccc}}"

aliases = {
    "BakedSDF": "BakedSDF",
    "BakedSDFtexturelss": "BakedSDF (textureless)",
    "BakedSDFTextureless": "BakedSDF (textureless)",
    "CAD": "CAD",
    "CADtex": "CAD (textured)",
    "Nerfacto": "Nerfacto",
    "NerfactoTextureless": "Nerfacto (textureless)",
    "gt": "GT",
}


def create_latex_notation_overall(path2folders, name_json="scores_bop19.json"):
    annotation_line = (
        "\multicolumn{2}{c|}{Method} & AR & $\mathrm{AR_{VSD}}$ & $\mathrm{AR_{MSPD}}$ & $\mathrm{AR_{MSSD}}$ "
        + END_ROW
        + HORIZONTAL_LINE
        + HORIZONTAL_LINE
    )
    latex_table = []
    latex_table.append(TABLE_HEADER_LINE_1)
    latex_table.append(TABLE_HEADER_LINE_2)
    latex_table.append(TABLE_HEADER_LINE_3)
    latex_table.append(annotation_line)
    results = {}

    contains = sorted(os.listdir(path2folders))
    for folder in contains:
        method = folder.split("_")[0]
        path2json = os.path.join(path2folders, folder, name_json)
        with open(path2json) as f:
            scores = json.load(f)
        results[method] = scores

    megapose_appearance = 0
    for method, scores in results.items():
        if "MegaPose" in method:
            method_name = method.replace("MegaPose", "")
            megapose_appearance += 1
        else:
            method_name = method

        AR = scores["bop19_average_recall"] * 100
        AR_VSD = scores["bop19_average_recall_vsd"] * 100
        AR_MSPD = scores["bop19_average_recall_mspd"] * 100
        AR_MSSD = scores["bop19_average_recall_mssd"] * 100

        if megapose_appearance == 1 and "MegaPose" in method:
            line_start = (
                "\multirow{6}{*}{{\\rotatebox[origin=c]{90}{MegaPose}}} & " + aliases[method_name]
            )
        else:
            line_start = "&" + aliases[method_name]

        line = f"{line_start} & {AR:.2f} & {AR_VSD:.2f} & {AR_MSPD:.2f} & {AR_MSSD:.2f} " + END_ROW
        if megapose_appearance == 6 and "MegaPose" in method:
            line += HORIZONTAL_LINE
        latex_table.append(line)

    caption = f"Evaluation results for all objects"
    end_1 = "\\end{tabular}"
    end_2 = f"\\caption{{{caption}}}\\label{{tab:AR_all}}"
    end_3 = "\\end{table}"
    latex_table.append(end_1)
    latex_table.append(end_2)
    latex_table.append(end_3)

    latex_table = "\n".join(latex_table)
    print(latex_table)


def create_latex_notation_per_object(path2result):
    results = {
        1: {},
        2: {},
        3: {},
        4: {},
        5: {},
        6: {},
        7: {},
        8: {},
    }
    # resuts -> object -> method -> error_metric -> ARvalue

    for method_folder in sorted(os.listdir(path2result)):
        method = method_folder.split("_")[0]
        # print(method)
        contains = sorted(os.listdir(os.path.join(path2result, method_folder)))
        for key in results.keys():
            results[key][method] = {}

        for error_metric_folder in contains:
            if ".json" in error_metric_folder:
                continue
            error_metric = error_metric_folder.split("_")[0].split("=")[1]
            if error_metric not in results[1][method].keys():
                for key in results.keys():
                    results[key][method][error_metric] = []

            if error_metric in ["mspd", "mssd", "vsd"]:
                inside = sorted(
                    os.listdir(os.path.join(path2result, method_folder, error_metric_folder))
                )
                for threshold_file in inside:
                    name_threshold = threshold_file.split("_")[0]
                    if name_threshold != "scores":
                        continue

                    # print(threshold_file)
                    path2json = os.path.join(
                        path2result, method_folder, error_metric_folder, threshold_file
                    )
                    with open(path2json) as f:
                        scores = json.load(f)
                    for object, score in scores["obj_recalls"].items():
                        # check if object with given metric exists

                        results[int(object)][method][error_metric].append(score)

    # print(results)
    # print(len(results[1]["MegaPoseBakedSDF"]["mspd"]))
    # print(len(results[1]["MegaPoseBakedSDF"]["mssd"]))
    # print(len(results[1]["MegaPoseBakedSDF"]["vsd"]))

    annotation_line = (
        "method & AR & $\\mathrm{AR_{VSD}}$ & $\\mathrm{AR_{MSPD}}$ & $\\mathrm{AR_{MSSD}}$ "
        + END_ROW
        + HORIZONTAL_LINE
    )

    f_line = (
        f"\multicolumn{{2}}{{c|}}{{\multirow{{2}}{{*}}{{{object}}}}}&  \multicolumn{{4}}{{c}}{{Overall}}"
        + END_ROW
    )
    s_line = (
        "\multicolumn{2}{c|}{} & AR & $\mathrm{AR_{VSD}}$ & $\mathrm{AR_{MSPD}}$ & $\mathrm{AR_{MSSD}}$ "
        + END_ROW
        + HORIZONTAL_LINE
    )

    for object in results.keys():
        object_name = LABELS[object]
        s_line = (
            "\multicolumn{2}{c|}{Method} & AR & $\mathrm{AR_{VSD}}$ & $\mathrm{AR_{MSPD}}$ & $\mathrm{AR_{MSSD}}$ "
            + END_ROW
            + HORIZONTAL_LINE
            + HORIZONTAL_LINE
        )
        latex_table = []
        latex_table.append(TABLE_HEADER_LINE_1)
        latex_table.append(TABLE_HEADER_LINE_2)
        latex_table.append(TABLE_HEADER_LINE_3)
        # latex_table.append(f_line)
        latex_table.append(s_line)
        megapose_appearance = 0
        for method in results[object].keys():
            if "MegaPose" in method:
                method_name = method.replace("MegaPose", "")
                megapose_appearance += 1
            else:
                method_name = method

            AR_VSD = np.mean(results[object][method]["vsd"]) * 100
            AR_MSPD = np.mean(results[object][method]["mspd"]) * 100
            AR_MSSD = np.mean(results[object][method]["mssd"]) * 100
            AR = np.mean([AR_VSD, AR_MSPD, AR_MSSD])
            if megapose_appearance == 1 and "MegaPose" in method:
                line_start = (
                    "\multirow{6}{*}{{\\rotatebox[origin=c]{90}{MegaPose}}} & "
                    + aliases[method_name]
                )
            else:
                line_start = "&" + aliases[method_name]
            line = (
                f"{line_start} & {AR:.2f} & {AR_VSD:.2f} & {AR_MSPD:.2f} & {AR_MSSD:.2f} " + END_ROW
            )
            if megapose_appearance == 6 and "MegaPose" in method:
                line += HORIZONTAL_LINE
            latex_table.append(line)

        caption = f"Evaluation results of {object_name}"
        end_1 = "\\end{tabular}"
        end_2 = f"\\caption{{{caption}}}\\label{{tab:AR_{object}}}"
        end_3 = "\\end{table}"
        latex_table.append(end_1)
        latex_table.append(end_2)
        latex_table.append(end_3)
        latex_table = "\n".join(latex_table)
        print(latex_table)
        print("\n")


if __name__ == "__main__":
    path2folders = os.path.join("Tags", "eval2")

    create_latex_notation_overall(path2folders)

    create_latex_notation_per_object(path2folders)

    # dict = {
    #     "obj_recalls": {
    #         "1": 0.1323529411764706,
    #         "2": 0.008238928939237899,
    #         "3": 0.004198740377886634,
    #         "4": 0.0,
    #         "5": 0.0,
    #         "6": 0.00429553264604811,
    #         "7": 0.0,
    #         "8": 0.0,
    #     },
    #     "recall": 0.017586474931986007,
    # }
    # recall = dict["recall"]
    # obj_recalls = dict["obj_recalls"]
    # obj_recalls = [obj_recall for _, obj_recall in obj_recalls.items()]

    # print(recall)
    # print(np.mean(obj_recalls))
