import pickle

with open("../Datasets/GUAVA_datasets/train/AIST++/sam3d_tmp/gBR_sBM_c02_d06_mBR3_ch01/00001.pkl", "rb") as f:
    data = pickle.load(f)

outputs = data["outputs"]   # list[dict] 或 []
print(len(outputs))
if outputs:
    print(outputs[0].keys())