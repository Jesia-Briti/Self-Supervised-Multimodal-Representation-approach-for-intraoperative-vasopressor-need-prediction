import os
import glob
import random
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
VP_INVENTORY_PATH = r"F:\Thesis_Dataset_corrected\Vasopressor_Processed_Data\Multimodal_Inventory_Summary.csv"

# Input Directories
VP_DIR = {
    'ppg': r"F:\Thesis_Dataset_corrected\Vasopressor_Processed_Data\ppg",
    'ecg': r"F:\Thesis_Dataset_corrected\Vasopressor_Processed_Data\ecg",
    'abp': r"F:\Thesis_Dataset_corrected\Vasopressor_Processed_Data\abp",
}

# >>> Point this at the NEW 1000-case control extraction output <<<
CTRL_DIR = {
    'ppg': r"F:\Control_Processed_Data_2500\ppg",
    'ecg': r"F:\Control_Processed_Data_2500\ecg",
    'abp': r"F:\Control_Processed_Data_2500\abp",
}

MASTER_OUT_DIR = r"F:\Master_Training_Data_Expanded_New_2"

# ==========================================
# 2. STRING-MATCHING LOGIC (HORIZON ALIGNMENT)
# ==========================================
# Mapping rule:  control_minute = 15 - vp_minute   (15-min control window)
# VP strings count BACKWARD from drug onset (0).
# Control strings count FORWARD from window start to 15m.
TARGET_OFFSETS = {
    'offset_3m':  {'vp_str': '-3m00s',  'ctrl_str': '12m00s', 'unified_str': '-3m00s'},
    'offset_4m':  {'vp_str': '-4m00s',  'ctrl_str': '11m00s', 'unified_str': '-4m00s'},
    'offset_5m':  {'vp_str': '-5m00s',  'ctrl_str': '10m00s', 'unified_str': '-5m00s'},
    'offset_6m':  {'vp_str': '-6m00s',  'ctrl_str': '9m00s',  'unified_str': '-6m00s'},
    'offset_7m':  {'vp_str': '-7m00s',  'ctrl_str': '8m00s',  'unified_str': '-7m00s'},
    'offset_10m': {'vp_str': '-10m00s', 'ctrl_str': '5m00s',  'unified_str': '-10m00s'},
    'offset_12m': {'vp_str': '-12m00s', 'ctrl_str': '3m00s',  'unified_str': '-12m00s'},  # NEW
    'offset_14m': {'vp_str': '-14m00s', 'ctrl_str': '1m00s',  'unified_str': '-14m00s'},  # NEW
    'offset_15m': {'vp_str': '-15m00s', 'ctrl_str': '0m00s',  'unified_str': '-15m00s'}
}

MODALITIES = ['ppg', 'ecg', 'abp']   # no capno
FILE_TYPES = ['wave', 'based']
SPLITS = ['train', 'val', 'test']

# Build directory tree for all horizons
for split in SPLITS:
    for mod in MODALITIES:
        for offset in TARGET_OFFSETS.keys():
            os.makedirs(os.path.join(MASTER_OUT_DIR, split, mod, offset), exist_ok=True)

# ==========================================
# 3. PATIENT-LEVEL SPLIT ENGINE
# ==========================================
def get_patient_split(case_ids, train_pct=0.8, val_pct=0.1):
    """Splits unique case_ids securely (80/10/10)."""
    random.seed(42)  # Strict lock for reproducibility
    cases = list(case_ids)
    random.shuffle(cases)

    n_total = len(cases)
    n_train = int(n_total * train_pct)
    n_val = int(n_total * val_pct)

    train_cases = set(cases[:n_train])
    val_cases = set(cases[n_train:n_train + n_val])
    test_cases = set(cases[n_train + n_val:])

    return train_cases, val_cases, test_cases

def slice_and_save_by_string(file_path, dest_dir, split_name, original_filename, is_control=False):
    """Extracts rows by exact time_offset string, unifies formatting, and saves one row per horizon."""
    if not os.path.exists(file_path):
        return 0

    try:
        df = pd.read_csv(file_path)
        slices_saved = 0

        for offset_name, strings in TARGET_OFFSETS.items():
            search_str = strings['ctrl_str'] if is_control else strings['vp_str']

            df_row = df[df['time_offset'] == search_str].copy()

            # Fail-Safe: segment dropped due to noise -> skip gracefully
            if df_row.empty:
                continue

            # Standardize
            df_row['time_offset'] = strings['unified_str']
            df_row['label_binary'] = 0 if is_control else 1
            if is_control:
                df_row['label'] = 0

            final_filename = f"{split_name}_{original_filename}"
            final_dest = os.path.join(dest_dir, offset_name, final_filename)

            df_row.to_csv(final_dest, index=False, quoting=0)
            slices_saved += 1

        return slices_saved
    except Exception as e:
        print(f"[!] Error processing {original_filename}: {e}")
        return 0

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def build_expanded_master_dataset():
    print("🚀 STARTING EXPANDED HORIZON COMPILATION (STRING-MATCHING MODE)")
    print(f"[*] Horizons: {list(TARGET_OFFSETS.keys())}")

    # --- A. VP CASES ---
    df_inv = pd.read_csv(VP_INVENTORY_PATH)
    df_vp_valid = df_inv[df_inv['category'] == 'ALL_THREE']
    vp_case_ids = df_vp_valid['case_id'].unique()
    vp_train, vp_val, vp_test = get_patient_split(vp_case_ids)

    # --- B. CONTROL CASES (new 2500-case set) ---
    ctrl_files = glob.glob(os.path.join(CTRL_DIR['ppg'], "*_wave.csv"))
    ctrl_case_ids = set()
    for f in ctrl_files:
        try:
            cid = int(os.path.basename(f).split('_')[1])
            ctrl_case_ids.add(cid)
        except Exception:
            pass
    ctrl_train, ctrl_val, ctrl_test = get_patient_split(ctrl_case_ids)

    print(f"\n📊 PATIENT-LEVEL SPLIT VERIFIED")
    print(f"   Train: {len(vp_train)} VP Cases, {len(ctrl_train)} Control Cases")
    print(f"   Val  : {len(vp_val)}  VP Cases, {len(ctrl_val)}  Control Cases")
    print(f"   Test : {len(vp_test)}  VP Cases, {len(ctrl_test)}  Control Cases")

    def get_split_name(cid, train_set, val_set, test_set):
        if cid in train_set: return 'train'
        if cid in val_set: return 'val'
        if cid in test_set: return 'test'
        return None

    total_slices = 0

    # --- C. PROCESS VASOPRESSOR DOSES ---
    print("\n⏳ Slicing Vasopressor Doses by Exact Time Strings...")
    for _, row in tqdm(df_vp_valid.iterrows(), total=len(df_vp_valid)):
        cid = int(row['case_id'])
        dose = int(row['dose_index'])
        split_name = get_split_name(cid, vp_train, vp_val, vp_test)
        if split_name is None:
            continue

        for mod in MODALITIES:
            for ftype in FILE_TYPES:
                filename = f"Case_{cid:04d}_dose_{dose}_{mod}_{ftype}.csv"
                src_path = os.path.join(VP_DIR[mod], filename)
                dest_base = os.path.join(MASTER_OUT_DIR, split_name, mod)
                total_slices += slice_and_save_by_string(src_path, dest_base, split_name, filename, is_control=False)

    # --- D. PROCESS CONTROL CASES ---
    print("\n⏳ Slicing Control Cases by Exact Time Strings...")
    for cid in tqdm(ctrl_case_ids):
        split_name = get_split_name(cid, ctrl_train, ctrl_val, ctrl_test)
        if split_name is None:
            continue

        for mod in MODALITIES:
            for ftype in FILE_TYPES:
                filename = f"Case_{cid:04d}_control_{mod}_{ftype}.csv"
                src_path = os.path.join(CTRL_DIR[mod], filename)
                dest_base = os.path.join(MASTER_OUT_DIR, split_name, mod)
                total_slices += slice_and_save_by_string(src_path, dest_base, split_name, filename, is_control=True)

    print(f"\n✅ MASTER MOVER COMPLETE!")
    print(f"🎉 Successfully verified and extracted {total_slices} unified temporal slices.")
    print(f"📁 Deep Learning directory created at: {MASTER_OUT_DIR}")

if __name__ == "__main__":
    build_expanded_master_dataset()