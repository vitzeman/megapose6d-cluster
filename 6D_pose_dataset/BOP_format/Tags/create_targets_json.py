import json
import os

PATHS2GTJSONS = ["test/000001/scene_gt.json"]


def create_targets_json():
    targets = []
    for path2gtjson in PATHS2GTJSONS:
        scene_id = path2gtjson.split("/")[-2]
        print(scene_id)
        scene_id = int(scene_id)
        with open(path2gtjson) as f:
            gt = json.load(f)
        for img_id, img_gt in gt.items():
            if len(img_gt) == 0:
                continue
            for gt_obj in img_gt:
                record = {
                    "im_id": int(img_id),
                    "inst_count": 1,  # Hardcoded the whole dataset was always captured with 1 instance per image
                    "obj_id": gt_obj["obj_id"],
                    "scene_id": scene_id,
                }
                targets.append(record)

    out_file = "test_targets_bop19.json"
    with open(out_file, "w") as f:
        print(f"Writing to {out_file}")
        json.dump(targets, f, indent=2)


if __name__ == "__main__":
    create_targets_json()
