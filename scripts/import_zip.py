import zipfile
from pathlib import Path

# dictionary for zip paths and folder names 
zip_key = {
    "school_lunch": "School Lunch.v2i.yolov11.zip",
    "tetra_pak": "tetra pak.v2i.yolov11.zip",
    "container": "Containers.v7i.yolov11.zip",
    "trash": "Trash Detection.v1i.yolov11.zip",
    "utensils": "Utensil.v1i.yolov11.zip",
    "condiments": "sauce.v1-v2.yolov11.zip"

}

for i, k in zip_key.items():

    output_folder = Path("data/raw") / i

    zip_path = Path.home() / "Downloads" / k

    with zipfile.ZipFile(zip_path, "r") as zip_file: 
        zip_file.extractall(output_folder)





