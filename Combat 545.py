#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationLearn, harmonizationApply

# Paths
data_dir = "/home/kexinguo/Combat545/data/train/train_vbm"
label_path = "/home/kexinguo/Combat545/data/train/official_site_class_labels.tsv"
memmap_path = "features_vbm_memmap.dat"
subject_id_path = "subject_ids_vbm.npy"


# In[3]:


bad_files = []

for fname in sorted(os.listdir(data_dir)):
    if fname.endswith(".npy"):
        path = os.path.join(data_dir, fname)
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as e:
            print(f"Cannot load {fname}: {e}")
            bad_files.append(fname)

print(f"\nTotal bad files: {len(bad_files)}")


# def detect_file_type(path):
#     with open(path, "rb") as f:
#         magic = f.read(4)
#     return magic
# 
# for fname in bad_files:
#     path = os.path.join(data_dir, fname)
#     try:
#         sig = detect_file_type(path)
#         print(f"{fname} → magic bytes: {sig}")
#     except Exception as e:
#         print(f"{fname} → unreadable: {e}")
# 

# In[7]:


meta_ids = meta["participant_id"].astype(str).tolist()
valid_ids = [sid for sid in meta_ids if sid in subject_ids]
np.array_equal(subject_ids, valid_ids)


# In[39]:


print(meta.columns.tolist())
print("Min value:", np.min(features[0]))
print("Max value:", np.max(features[0]))
print("Mean value:", np.mean(features[0]))
print("Std value:", np.std(features[0]))


# In[8]:


# Ensure correct dtype
meta['participant_id'] = meta['participant_id'].astype(str)

# Align rows in the same order as features
meta_sub = meta.set_index("participant_id").loc[subject_ids]
batch = meta_sub['siteXacq'].values

#print(meta_sub)
print(batch)
#print(meta_sub.index.tolist())  # should match subject_ids
#print(meta_sub["siteXacq"].values)  # your batch variable for ComBat


# In[9]:


# Create covars DataFrame with site as 'SITE'
covars = pd.DataFrame({
    "SITE": meta_sub["siteXacq"].values
}, index=meta_sub.index)

print(covars["SITE"].value_counts())


# covars = pd.DataFrame({
#     "SITE": meta_sub["siteXacq"].values
# }, index=meta_sub.index)
# 
# print(covars["SITE"].value_counts())
# features_filtered_combat = combat_by_block(features_subset, covars, block_size=30000)
# print("Harmonized shape:", features_filtered_combat.shape)

# In[2]:


def combat_by_block(features, covars, block_size):
    n_subjects, n_voxels = features.shape
    harmonized = np.zeros_like(features)

    for start in range(0, n_voxels, block_size):
        end = min(start + block_size, n_voxels)
        print(f"Processing block {start}:{end}")

        block = features[:, start:end]
        try:
            model, _ = harmonizationLearn(block, covars, ref_batch="3")
            harmonized[:, start:end] = harmonizationApply(block, covars, model)
        except Exception as e:
            print(f"Skipped block {start}:{end} due to error: {e}")
            harmonized[:, start:end] = block  # fallback: leave unharmonized

    return harmonized


# # Apply ComBat

# In[11]:


# Calculate std across subjects for each voxel
stds = np.std(features, axis=0)

# Keep only features with non-trivial variation
threshold = 0.05
mask = stds > threshold
features_filtered = features[:, mask]

print("Original shape:", features.shape)
print("Filtered shape:", features_filtered.shape)


# In[12]:


features_filtered_combat = combat_by_block(features_filtered, covars, block_size=30000)
print("Harmonized shape:", features_filtered_combat.shape)


# In[13]:


print("NaNs:", np.isnan(features_filtered_combat).sum())
#print("Infs:", np.isinf(features_combat).sum())


# In[14]:


features_combat = np.zeros_like(features)
features_combat[:, mask] = features_filtered_combat
np.save("features_combat_vbm.npy", features_combat)
# to reload
## features_combat = np.memmap("features_vbm_memmap.dat", dtype='float32', shape=(3227, 2122945), mode='r')


# # Make two boxplots: before and after

# In[15]:


import matplotlib.pyplot as plt
import numpy as np

# Compute subject-wise means
subject_means_before = features.mean(axis=1)
subject_means_after = features_combat.mean(axis=1)
sites = np.array(meta_sub["siteXacq"])

# Pick top N most common sites for comparison
site_counts = meta_sub["siteXacq"].value_counts()
top_sites = site_counts.index[:6]  # Change to 8 or 10 if you want more

# Group subject means by site
means_by_site_before = [subject_means_before[sites == s] for s in top_sites]
means_by_site_after = [subject_means_after[sites == s] for s in top_sites]

# Shared y-axis range
ymin = min(subject_means_before.min(), subject_means_after.min())
ymax = max(subject_means_before.max(), subject_means_after.max())

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

axes[0].boxplot(means_by_site_before, labels=[f"Site {s}" for s in top_sites])
axes[0].set_title("Subject Means Before ComBat")
axes[0].set_ylabel("Mean Voxel Intensity")
axes[0].set_ylim(ymin, ymax)

axes[1].boxplot(means_by_site_after, labels=[f"Site {s}" for s in top_sites])
axes[1].set_title("Subject Means After ComBat")
axes[1].set_ylim(ymin, ymax)

plt.tight_layout()
plt.show()


# In[16]:


def summarize_data(name, X):
    print(f"\n Summary of {name}:")
    print(f"  Global min: {np.min(X):.4f}")
    print(f"  Global max: {np.max(X):.4f}")
    print(f"  Mean across all subjects: {np.mean(X):.4f}")
    print(f"  Std across all subjects: {np.std(X):.4f}")
    

summarize_data("Before ComBat", features)
summarize_data("After ComBat", features_combat)


# ## check the shape and number of voxels

# In[16]:


print(features_combat.shape)


# In[17]:


import nibabel as nib
import numpy as np

# Load one sample VBM volume (any subject, pre- or post-harmonization is fine)
example_vbm = nib.load("/home/kexinguo/Combat545/data/cat12vbm_space-MNI152_desc-gm_TPM.nii.gz")
vbm_shape = example_vbm.shape
print("3D shape of VBM image:", vbm_shape)


# In[18]:


expected_voxels = np.prod(vbm_shape)
print("Expected flattened size:", expected_voxels)


# In[19]:


features_3d = features_combat.reshape(-1, 121, 145, 121)
print("Reshaped shape:", features_3d.shape)
np.save("features_3d_combat_vbm.npy", features_3d)


# In[22]:


import matplotlib.pyplot as plt

sample = features_3d[1]  # shape: (121, 145, 121)
mid_slice = sample[:, :, sample.shape[2] // 2]

plt.imshow(mid_slice, cmap='gray')
plt.title("Mid-Slice of Harmonized Volume")
plt.axis("off")
plt.show()


# # val

# In[66]:


# ---------------- VALIDATION SET ----------------
val_data_dir = "/home/kexinguo/Combat545/data/val/val_vbm"
val_label_path = "/home/kexinguo/Combat545/data/val/official_site_class_labels.tsv"
val_memmap_path = "features_val_vbm_memmap.dat"
val_subject_id_path = "subject_ids_val_vbm.npy"

val_meta = pd.read_csv(val_label_path, sep="\t")
val_meta['participant_id'] = val_meta['participant_id'].astype(str)

val_npy_files = sorted([f for f in os.listdir(val_data_dir) if f.endswith(".npy")])
val_n_subjects = len(val_npy_files)
val_sample_shape = np.load(os.path.join(val_data_dir, val_npy_files[0])).shape
val_n_voxels = np.prod(val_sample_shape)

val_features = np.memmap(val_memmap_path, dtype='float32', mode='w+', shape=(val_n_subjects, val_n_voxels))
val_subject_ids = []

for i, fname in enumerate(val_npy_files):
    subj_id = fname.split("_")[0].split("-")[1]
    val_subject_ids.append(subj_id)
    arr = np.load(os.path.join(val_data_dir, fname)).astype(np.float32)
    val_features[i] = arr.flatten()

val_features.flush()
np.save(val_subject_id_path, val_subject_ids)
val_meta_sub = val_meta.set_index("participant_id").loc[val_subject_ids]

site_counts = val_meta_sub["siteXacq"].value_counts()
singleton_sites = site_counts[site_counts <= 1].index.tolist()

val_meta_sub["SITE_COMBAT"] = val_meta_sub["siteXacq"].apply(
    lambda x: 999 if x in singleton_sites else x
)
val_covars = pd.DataFrame({"SITE": val_meta_sub["SITE_COMBAT"].values}, index=val_meta_sub.index)
#val_covars = pd.DataFrame({"SITE": val_meta_sub["siteXacq"].values}, index=val_meta_sub.index)

print(val_meta.columns.tolist())
print("Min value:", np.min(val_features[0]))
print("Max value:", np.max(val_features[0]))
print("Mean value:", np.mean(val_features[0]))
print("Std value:", np.std(val_features[0]))
print(val_covars["SITE"].value_counts())


# In[62]:


val_stds = np.std(val_features, axis=0)
val_mask = val_stds > 0.05  # Use same threshold as training
val_features_filtered = val_features[:, val_mask]

print(val_features.shape)
print(val_features_filtered.shape)
print("NaNs:", np.isnan(val_features_filtered).sum())


# In[63]:


val_features_filtered_combat = combat_by_block(val_features_filtered, val_covars, block_size=30000)
print("NaNs:", np.isnan(val_features_filtered_combat).sum())


# In[64]:


val_features_combat = np.zeros_like(val_features)
val_features_combat[:, val_mask] = val_features_filtered_combat

np.save("features_val_combat_vbm.npy", val_features_combat)


# In[65]:


import matplotlib.pyplot as plt
import numpy as np

# Compute subject-wise means
val_subject_means_before = val_features.mean(axis=1)
val_subject_means_after = val_features_combat.mean(axis=1)
val_sites = np.array(val_meta_sub["siteXacq"])

# Pick top N most common sites for comparison
val_site_counts = val_meta_sub["siteXacq"].value_counts()
val_top_sites = val_site_counts.index[:6]  # Change to more if needed

# Group subject means by site
val_means_by_site_before = [val_subject_means_before[val_sites == s] for s in val_top_sites]
val_means_by_site_after = [val_subject_means_after[val_sites == s] for s in val_top_sites]

# Shared y-axis range
ymin = min(val_subject_means_before.min(), val_subject_means_after.min())
ymax = max(val_subject_means_before.max(), val_subject_means_after.max())

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

axes[0].boxplot(val_means_by_site_before, labels=[f"Site {s}" for s in val_top_sites])
axes[0].set_title("Val Subject Means Before ComBat")
axes[0].set_ylabel("Mean Voxel Intensity")
axes[0].set_ylim(ymin, ymax)

axes[1].boxplot(val_means_by_site_after, labels=[f"Site {s}" for s in val_top_sites])
axes[1].set_title("Val Subject Means After ComBat")
axes[1].set_ylim(ymin, ymax)

plt.tight_layout()
plt.show()


# In[ ]:


# Reshape back to 3D and save
val_features_3d = val_features_combat.reshape(-1, *val_sample_shape)  # shape: (val_n_subjects, 121, 145, 121)
print("Reshaped shape:", val_features_3d.shape)
np.save("features_3d_val_combat_vbm.npy", val_features_3d)

