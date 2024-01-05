import csv

FIRST_LINE = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
M2MM = 1000


def convert_m2mm_csv(csv_file):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        lines = list(reader)
    new_lines = []
    for e, line in enumerate(lines):
        if e == 0:
            new_lines.append(FIRST_LINE)
            continue

        new_line = []
        new_line.append(line[0])
        new_line.append(line[1])
        new_line.append(line[2])
        new_line.append(line[3])

        # 5th element is R
        R_elems = line[4].replace(",", "")
        new_line.append(R_elems)
        # Convert meter to millimeter
        # print(line[5])

        elems = line[5].split(", ")
        new_elems = [float(e) * M2MM for e in elems]
        # print(new_elems)
        stripped_T = str(new_elems).strip("[]").replace(",", "")
        new_line.append(stripped_T)

        new_line.append(line[6])
        new_lines.append(new_line)

    new_csv_file = "Tags_" + csv_file.replace(".csv", "_mm.csv")
    with open(new_csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(new_lines)


if __name__ == "__main__":
    csv_files = [
        "megapose_meshes_BakedSDF.csv",
        "megapose_meshes_BakedSDF_textureless.csv",
        "megapose_CAD_alligned.csv",
        "megaposeCAD_textured.csv",
        "megaposeNerfacto.csv",
        "megapose_Nerfacto_cleared_scaled_alligned_textureless.csv",
    ]
    for csv_file in csv_files:
        convert_m2mm_csv(csv_file)

    csv_files_mms = [
        "Tags_megapose_meshes_BakedSDF_mm.csv",
        "Tags_megapose_meshes_BakedSDF_textureless_mm.csv",
        "Tags_megapose_CAD_alligned_mm.csv",
        "Tags_megaposeCAD_textured_mm.csv",
        "Tags_megaposeNerfacto_mm.csv",
        "Tags_megapose_Nerfacto_cleared_scaled_alligned_textureless_mm.csv",
    ]
