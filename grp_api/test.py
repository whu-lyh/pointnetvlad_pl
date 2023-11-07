import numpy as np
from feature import PointNetVladFeature
from gpr.dataloader import PittsLoader
from gpr.evaluation import get_recall
from gpr.tools import save_feature_for_submission
from tqdm import tqdm

from option import get_options

hparams = get_options()

# Test Data Loader, change to your datafolder
pitts_loader = PittsLoader("/lyh/GPR_competition/UGV/TEST")
# pitts_loader = PittsLoader('../datasets/gpr_pitts_sample/')

# Point cloud conversion and feature extractor
PNV_fea = PointNetVladFeature("/workspace/pointnetvlad_pl/logs/ckpts/ugv_pr/epoch=99.ckpt", hparams.prefixes_to_ignore)

# feature extraction step by step
feature_ref = []
for idx in tqdm(range(len(pitts_loader)), desc = 'comp. fea.'):
    pcd_ref = []   
    pcd_ = pitts_loader[idx-1]['pcd']
    pcd_ref.append(pcd_)
    feature_ref.append(PNV_fea.infer_data(pcd_ref))

# evaluate recall
feature_ref = np.array(feature_ref)
topN_recall, one_percent_recall = get_recall(feature_ref, feature_ref)

print("topN_recall", topN_recall)
print("one_percent_recall", one_percent_recall)
save_feature_for_submission('./submissions/pnv.npy', feature_ref)
