from configparser import ConfigParser
from pathlib import Path

def create_config_file(output_path: str = "lung.conf") -> None:
    """
    Create a minimal configuration file for the LIDC-IDRI preprocessing pipeline.

    The file stores (i) paths to raw DICOM data and output folders, and
    (ii) a small set of preprocessing hyperparameters used by downstream scripts.
    Paths are relative to the working directory; adjust as needed.
    """
    cfg = ConfigParser()

    # I. dataset preparation
    cfg["prepare_dataset"] = {
        "LIDC_DICOM_PATH": "./LIDC-IDRI",
        "MASK_PATH": "./data/Mask",
        "IMAGE_PATH": "./data/Image",
        "CLEAN_PATH_IMAGE": "./data/Clean/Image",
        "CLEAN_PATH_MASK": "./data/Clean/Mask",
        "META_PATH": "./data/Meta/",
        "Mask_Threshold": "8",  # discard tiny masks (area < 8 px)
    }

    # II. pylidc / consensus settings
    cfg["pylidc"] = {
        "confidence_level": "0.5",  # minimum annotator agreement
        "padding_size": "512",      # in-plane padding (pixels)
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        cfg.write(f)

    print(f"Config written to: {out.resolve()}")


if __name__ == "__main__":
    create_config_file()



