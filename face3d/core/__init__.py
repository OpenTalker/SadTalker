from core.BFM09Model import BFM09ReconModel
from scipy.io import loadmat


def get_recon_model(model='bfm09', **kargs):
    if model == 'bfm09':
        model_path = '/apdcephfs_cq2/share_1290939/shadowcun/documents/BFM_Fitting/BFM09_model_info.mat'
        model_dict = loadmat(model_path)
        recon_model = BFM09ReconModel(model_dict, **kargs)
        return recon_model
    else:
        raise NotImplementedError()
