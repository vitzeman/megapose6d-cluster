import os
import json

HORIZONTAL_LINE = "\\hline"
END_ROW = "\\\\"


def create_latex_notation(path2folders, name_json="scores_bop19.json"):
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


if __name__ == "__main__":
    path2folders = os.path.join("Tags", "eval")

    create_latex_notation(path2folders)
