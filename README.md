# MDE_SFM_3D
Use ColMAP sparse SFM result+ depth anything v2 generated metric estimation to do 3D reconstruction

# === Modify these paths before running ===
colmap_model_path = "/Users/helen/Documents/3DGS_data/small_spiddy_run2/0" ## colmap sparse SFM folder
<img width="667" alt="image" src="https://github.com/user-attachments/assets/157785c7-4694-46c5-8ef7-cc1a29561d16" />

npy_dir = "/Users/helen/Documents/3D/depth_anything_v2/metric_depth/small_spiddy/output" ## Depth Anything v2 metric depth results as npy files
<img width="755" alt="image" src="https://github.com/user-attachments/assets/77287715-42ff-4e15-8062-3bc144f8ffc8" />



if __name__ == "__main__":
    align_mde_to_colmap_with_icp(colmap_model_path, npy_dir)

<img width="1405" alt="131743753853_ pic_hd" src="https://github.com/user-attachments/assets/f02314ce-d6e9-40f5-ab76-ba331f9932b6" />
