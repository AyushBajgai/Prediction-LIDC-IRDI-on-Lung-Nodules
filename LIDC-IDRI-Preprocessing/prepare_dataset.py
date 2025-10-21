import os
import warnings
from pathlib import Path
from configparser import ConfigParser
from statistics import median_high

import numpy as np
import pandas as pd
import pylidc as pl
from PIL import Image
from tqdm import tqdm
from pylidc.utils import consensus

from utils import is_dir_path, segment_lung

warnings.filterwarnings("ignore")

# ============================================================
# Compatibility fix for numpy >= 1.24 (np.int was deprecated)
# ============================================================
if not hasattr(np, "int"):
    np.int = int


# ============================================================
# Function: read and parse configuration file (lung.conf)
# ============================================================
def load_cfg(cfg_file: str = "lung.conf"):
    parser = ConfigParser()
    parser.read(cfg_file)

    # Retrieve dataset paths from the config file
    dicom_dir = is_dir_path(parser.get("prepare_dataset", "LIDC_DICOM_PATH"))
    mask_dir = is_dir_path(parser.get("prepare_dataset", "MASK_PATH"))
    img_dir = is_dir_path(parser.get("prepare_dataset", "IMAGE_PATH"))
    clean_img_dir = is_dir_path(parser.get("prepare_dataset", "CLEAN_PATH_IMAGE"))
    clean_mask_dir = is_dir_path(parser.get("prepare_dataset", "CLEAN_PATH_MASK"))
    meta_dir = is_dir_path(parser.get("prepare_dataset", "META_PATH"))

    # Retrieve hyperparameters
    mask_thresh = parser.getint("prepare_dataset", "Mask_Threshold")
    c_level = parser.getfloat("pylidc", "confidence_level")
    pad = parser.getint("pylidc", "padding_size")

    # Return a structured dictionary for easier access
    return {
        "dicom": Path(dicom_dir),
        "mask": Path(mask_dir),
        "img": Path(img_dir),
        "clean_img": Path(clean_img_dir),
        "clean_mask": Path(clean_mask_dir),
        "meta": Path(meta_dir),
        "mask_thresh": mask_thresh,
        "c_level": c_level,
        "pad": pad,
    }


# ============================================================
# Utility: ensure that required directories exist
# ============================================================
def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility: compute median malignancy and assign cancer label
# median_high() ensures that ties are rounded toward the higher value
# ============================================================
def malignancy_label(nodule):
    vals = [ann.malignancy for ann in nodule]
    m = median_high(vals)
    if m > 3:
        return m, True
    if m < 3:
        return m, False
    return m, "Ambiguous"


# ============================================================
# Utility: append one metadata record to the DataFrame
# ============================================================
def add_meta_row(meta_df: pd.DataFrame, row_list):
    cols = [
        "patient_id",
        "nodule_no",
        "slice_no",
        "original_image",
        "mask_image",
        "malignancy",
        "is_cancer",
        "is_clean",
    ]
    s = pd.Series(row_list, index=cols)
    return pd.concat([meta_df, s], ignore_index=True)


# ============================================================
# Utility: list valid patient directories (ignore hidden files)
# ============================================================
def list_patients(dicom_root: Path):
    return sorted([f.name for f in dicom_root.iterdir() if not f.name.startswith(".")])


# ============================================================
# Utility: generate zero-padded file name prefixes ("000"–"999")
# ============================================================
def file_prefix_pool(n=1000):
    return [str(i).zfill(3) for i in range(n)]


# ============================================================
# Utility: create subfolders for a single patient
# ============================================================
def patient_dirs(base_img: Path, base_mask: Path, pid: str):
    di = base_img / pid
    dm = base_mask / pid
    di.mkdir(parents=True, exist_ok=True)
    dm.mkdir(parents=True, exist_ok=True)
    return di, dm


# ============================================================
# Class: DatasetBuilder — main dataset construction pipeline
# ============================================================
class DatasetBuilder:
    def __init__(self, cfg: dict):
        self.paths = cfg
        self.mask_threshold = cfg["mask_thresh"]
        self.c_level = cfg["c_level"]
        pad = cfg["pad"]
        # padding format required by pylidc consensus()
        self.pad_tuple = [(pad, pad), (pad, pad), (0, 0)]
        # initialize metadata table
        self.meta = pd.DataFrame(
            columns=[
                "patient_id",
                "nodule_no",
                "slice_no",
                "original_image",
                "mask_image",
                "malignancy",
                "is_cancer",
                "is_clean",
            ]
        )
        # generate prefix sequence for file naming
        self.prefix = file_prefix_pool(1000)

    # ========================================================
    # Process nodule-positive scans (patients with nodules)
    # ========================================================
    def _process_nodules(self, vol, nodules_annotation, pid, save_dir_img, save_dir_mask):
        for nid, nodule in enumerate(nodules_annotation):
            # Combine four radiologist annotations using consensus
            mask, cbbox, _ = consensus(nodule, self.c_level, self.pad_tuple)
            vol_roi = vol[cbbox]
            malignancy, label = malignancy_label(nodule)

            for z in range(mask.shape[2]):
                m_slice = mask[:, :, z]
                if np.sum(m_slice) <= self.mask_threshold:
                    continue

                # Apply lung segmentation to each slice
                img_slice = segment_lung(vol_roi[:, :, z])
                img_slice[img_slice == -0] = 0  # fix potential float -0

                # naming convention kept identical for downstream compatibility
                n_name = f"{pid[-4:]}_NI{self.prefix[nid]}_slice{self.prefix[z]}"
                m_name = f"{pid[-4:]}_MA{self.prefix[nid]}_slice{self.prefix[z]}"

                # append to metadata
                self.meta = add_meta_row(
                    self.meta,
                    [pid[-4:], nid, self.prefix[z], n_name, m_name, malignancy, label, False],
                )

                # save image and mask as numpy arrays
                np.save(save_dir_img / n_name, img_slice)
                np.save(save_dir_mask / m_name, m_slice)

    # ========================================================
    # Process “clean” (nodule-free) patients — up to 50 slices
    # ========================================================
    def _process_clean(self, vol, pid, clean_img_dir, clean_mask_dir):
        clean_img_dir.mkdir(parents=True, exist_ok=True)
        clean_mask_dir.mkdir(parents=True, exist_ok=True)

        for z in range(min(vol.shape[2], 51)):
            img_slice = segment_lung(vol[:, :, z])
            img_slice[img_slice == -0] = 0
            zero_mask = np.zeros_like(img_slice)

            n_name = f"{pid[-4:]}_CN001_slice{self.prefix[z]}"
            m_name = f"{pid[-4:]}_CM001_slice{self.prefix[z]}"

            self.meta = add_meta_row(
                self.meta, [pid[-4:], z, self.prefix[z], n_name, m_name, 0, False, True]
            )

            np.save(clean_img_dir / n_name, img_slice)
            np.save(clean_mask_dir / m_name, zero_mask)

    # ========================================================
    # Fetch a single patient’s scan and annotation info
    # ========================================================
    def _fetch_scan(self, pid: str):
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        if scan is None:
            return None, None, None
        nods = scan.cluster_annotations()
        vol = scan.to_volume()
        return scan, nods, vol

    # ========================================================
    # Main build routine — iterate through all patients
    # ========================================================
    def build(self):
        # ensure all base directories exist
        ensure_dirs(
            [
                self.paths["img"],
                self.paths["mask"],
                self.paths["clean_img"],
                self.paths["clean_mask"],
                self.paths["meta"],
            ]
        )

        patients = list_patients(self.paths["dicom"])

        for pid in tqdm(patients, desc="Patients"):
            scan, nods, vol = self._fetch_scan(pid)
            if scan is None:
                print(f"[WARN] Skipped {pid}: scan not found.")
                continue

            print(f"Patient ID: {pid}  Dicom Shape: {vol.shape}  Nodules: {len(nods)}")

            out_img_dir, out_mask_dir = patient_dirs(self.paths["img"], self.paths["mask"], pid)

            if len(nods) > 0:
                self._process_nodules(vol, nods, pid, out_img_dir, out_mask_dir)
            else:
                print(f"Clean Dataset {pid}")
                self._process_clean(
                    vol, pid, self.paths["clean_img"] / pid, self.paths["clean_mask"] / pid
                )

        # save metadata as CSV
        meta_csv = self.paths["meta"] / "meta_info.csv"
        self.meta.to_csv(meta_csv, index=False)
        print(f"Saved metadata -> {meta_csv.resolve()}")


# ============================================================
# Script entry point
# ============================================================
if __name__ == "__main__":
    cfg = load_cfg("lung.conf")
    builder = DatasetBuilder(cfg)
    builder.build()




