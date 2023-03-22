"""This script is to generate skin attention mask for Deep3DFaceRecon_pytorch
"""

import math
import numpy as np
import os
import cv2

class GMM:
    def __init__(self, dim, num, w, mu, cov, cov_det, cov_inv):
        self.dim = dim # feature dimension
        self.num = num # number of Gaussian components
        self.w = w # weights of Gaussian components (a list of scalars)
        self.mu= mu # mean of Gaussian components (a list of 1xdim vectors)
        self.cov = cov # covariance matrix of Gaussian components (a list of dimxdim matrices)
        self.cov_det = cov_det # pre-computed determinet of covariance matrices (a list of scalars)
        self.cov_inv = cov_inv # pre-computed inverse covariance matrices (a list of dimxdim matrices)

        self.factor = [0]*num
        for i in range(self.num):
            self.factor[i] = (2*math.pi)**(self.dim/2) * self.cov_det[i]**0.5
        
    def likelihood(self, data):
        assert(data.shape[1] == self.dim)
        N = data.shape[0]
        lh = np.zeros(N)

        for i in range(self.num):
            data_ = data - self.mu[i]

            tmp = np.matmul(data_,self.cov_inv[i]) * data_
            tmp = np.sum(tmp,axis=1)
            power = -0.5 * tmp

            p = np.array([math.exp(power[j]) for j in range(N)])
            p = p/self.factor[i]
            lh += p*self.w[i]
        
        return lh


def _rgb2ycbcr(rgb):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = rgb.shape
    rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:, 0] += 16.
    ycbcr[:, 1:] += 128.
    return ycbcr.reshape(shape)


def _bgr2ycbcr(bgr):
    rgb = bgr[..., ::-1]
    return _rgb2ycbcr(rgb)


gmm_skin_w = [0.24063933, 0.16365987, 0.26034665, 0.33535415]
gmm_skin_mu = [np.array([113.71862, 103.39613, 164.08226]),
                np.array([150.19858, 105.18467, 155.51428]),
                np.array([183.92976, 107.62468, 152.71820]),
                np.array([114.90524, 113.59782, 151.38217])]
gmm_skin_cov_det = [5692842.5, 5851930.5, 2329131., 1585971.]
gmm_skin_cov_inv = [np.array([[0.0019472069, 0.0020450759, -0.00060243998],[0.0020450759, 0.017700525, 0.0051420014],[-0.00060243998, 0.0051420014, 0.0081308950]]),
                    np.array([[0.0027110141, 0.0011036990, 0.0023122299],[0.0011036990, 0.010707724, 0.010742856],[0.0023122299, 0.010742856, 0.017481629]]),
                    np.array([[0.0048026871, 0.00022935172, 0.0077668377],[0.00022935172, 0.011729696, 0.0081661865],[0.0077668377, 0.0081661865, 0.025374353]]),
                    np.array([[0.0011989699, 0.0022453172, -0.0010748957],[0.0022453172, 0.047758564, 0.020332102],[-0.0010748957, 0.020332102, 0.024502251]])]

gmm_skin = GMM(3, 4, gmm_skin_w, gmm_skin_mu, [], gmm_skin_cov_det, gmm_skin_cov_inv)

gmm_nonskin_w = [0.12791070, 0.31130761, 0.34245777, 0.21832393]
gmm_nonskin_mu = [np.array([99.200851, 112.07533, 140.20602]),
                    np.array([110.91392, 125.52969, 130.19237]),
                    np.array([129.75864, 129.96107, 126.96808]),
                    np.array([112.29587, 128.85121, 129.05431])]
gmm_nonskin_cov_det = [458703648., 6466488., 90611376., 133097.63]
gmm_nonskin_cov_inv = [np.array([[0.00085371657, 0.00071197288, 0.00023958916],[0.00071197288, 0.0025935620, 0.00076557708],[0.00023958916, 0.00076557708, 0.0015042332]]),
                    np.array([[0.00024650150, 0.00045542428, 0.00015019422],[0.00045542428, 0.026412144, 0.018419769],[0.00015019422, 0.018419769, 0.037497383]]),
                    np.array([[0.00037054974, 0.00038146760, 0.00040408765],[0.00038146760, 0.0085505722, 0.0079136286],[0.00040408765, 0.0079136286, 0.010982352]]),
                    np.array([[0.00013709733, 0.00051228428, 0.00012777430],[0.00051228428, 0.28237113, 0.10528370],[0.00012777430, 0.10528370, 0.23468947]])]

gmm_nonskin = GMM(3, 4, gmm_nonskin_w, gmm_nonskin_mu, [], gmm_nonskin_cov_det, gmm_nonskin_cov_inv)

prior_skin = 0.8
prior_nonskin = 1 - prior_skin


# calculate skin attention mask
def skinmask(imbgr):
    im = _bgr2ycbcr(imbgr)

    data = im.reshape((-1,3))

    lh_skin = gmm_skin.likelihood(data)
    lh_nonskin = gmm_nonskin.likelihood(data)

    tmp1 = prior_skin * lh_skin
    tmp2 = prior_nonskin * lh_nonskin
    post_skin = tmp1 / (tmp1+tmp2) # posterior probability

    post_skin = post_skin.reshape((im.shape[0],im.shape[1]))

    post_skin = np.round(post_skin*255)
    post_skin = post_skin.astype(np.uint8)
    post_skin = np.tile(np.expand_dims(post_skin,2),[1,1,3]) # reshape to H*W*3

    return post_skin


def get_skin_mask(img_path):
    print('generating skin masks......')
    names = [i for i in sorted(os.listdir(
        img_path)) if 'jpg' in i or 'png' in i or 'jpeg' in i or 'PNG' in i]
    save_path = os.path.join(img_path, 'mask')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    for i in range(0, len(names)):
        name = names[i]
        print('%05d' % (i), ' ', name)
        full_image_name = os.path.join(img_path, name)
        img = cv2.imread(full_image_name).astype(np.float32)
        skin_img = skinmask(img)
        cv2.imwrite(os.path.join(save_path, name), skin_img.astype(np.uint8))
