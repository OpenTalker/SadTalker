"""This script defines the face reconstruction model for Deep3DFaceRecon_pytorch
"""

import numpy as np
import torch
from src.face3d.models.base_model import BaseModel
from src.face3d.models import networks
from src.face3d.models.bfm import ParametricFaceModel
from src.face3d.models.losses import perceptual_loss, photo_loss, reg_loss, reflectance_loss, landmark_loss
from src.face3d.util import util 
from src.face3d.util.nvdiffrast import MeshRenderer
# from src.face3d.util.preprocess import estimate_norm_torch

import trimesh
from scipy.io import savemat

class FaceReconModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        """  Configures options specific for CUT model
        """
        # net structure and parameters
        parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
        parser.add_argument('--init_path', type=str, default='./checkpoints/init_model/resnet50-0676ba61.pth')
        parser.add_argument('--use_last_fc', type=util.str2bool, nargs='?', const=True, default=False, help='zero initialize the last fc')
        parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
        parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

        # renderer parameters
        parser.add_argument('--focal', type=float, default=1015.)
        parser.add_argument('--center', type=float, default=112.)
        parser.add_argument('--camera_d', type=float, default=10.)
        parser.add_argument('--z_near', type=float, default=5.)
        parser.add_argument('--z_far', type=float, default=15.)

        if is_train:
            # training parameters
            parser.add_argument('--net_recog', type=str, default='r50', choices=['r18', 'r43', 'r50'], help='face recog network structure')
            parser.add_argument('--net_recog_path', type=str, default='checkpoints/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth')
            parser.add_argument('--use_crop_face', type=util.str2bool, nargs='?', const=True, default=False, help='use crop mask for photo loss')
            parser.add_argument('--use_predef_M', type=util.str2bool, nargs='?', const=True, default=False, help='use predefined M for predicted face')

            
            # augmentation parameters
            parser.add_argument('--shift_pixs', type=float, default=10., help='shift pixels')
            parser.add_argument('--scale_delta', type=float, default=0.1, help='delta scale factor')
            parser.add_argument('--rot_angle', type=float, default=10., help='rot angles, degree')

            # loss weights
            parser.add_argument('--w_feat', type=float, default=0.2, help='weight for feat loss')
            parser.add_argument('--w_color', type=float, default=1.92, help='weight for loss loss')
            parser.add_argument('--w_reg', type=float, default=3.0e-4, help='weight for reg loss')
            parser.add_argument('--w_id', type=float, default=1.0, help='weight for id_reg loss')
            parser.add_argument('--w_exp', type=float, default=0.8, help='weight for exp_reg loss')
            parser.add_argument('--w_tex', type=float, default=1.7e-2, help='weight for tex_reg loss')
            parser.add_argument('--w_gamma', type=float, default=10.0, help='weight for gamma loss')
            parser.add_argument('--w_lm', type=float, default=1.6e-3, help='weight for lm loss')
            parser.add_argument('--w_reflc', type=float, default=5.0, help='weight for reflc loss')

        opt, _ = parser.parse_known_args()
        parser.set_defaults(
                focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
            )
        if is_train:
            parser.set_defaults(
                use_crop_face=True, use_predef_M=False
            )
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        
        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer']

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=self.isTrain, default_name=opt.bfm_model
        )
        
        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center)
        )

        if self.isTrain:
            self.loss_names = ['all', 'feat', 'color', 'lm', 'reg', 'gamma', 'reflc']

            self.net_recog = networks.define_net_recog(
                net_recog=opt.net_recog, pretrained_path=opt.net_recog_path
                )
            # loss func name: (compute_%s_loss) % loss_name
            self.compute_feat_loss = perceptual_loss
            self.comupte_color_loss = photo_loss
            self.compute_lm_loss = landmark_loss
            self.compute_reg_loss = reg_loss
            self.compute_reflc_loss = reflectance_loss

            self.optimizer = torch.optim.Adam(self.net_recon.parameters(), lr=opt.lr)
            self.optimizers = [self.optimizer]
            self.parallel_names += ['net_recog']
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs'].to(self.device) 
        self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        self.gt_lm = input['lms'].to(self.device)  if 'lms' in input else None
        self.trans_m = input['M'].to(self.device) if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None

    def forward(self, output_coeff, device):
        self.facemodel.to(device)
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(output_coeff)
        self.pred_mask, _, self.pred_face = self.renderer(
            self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)
        
        self.pred_coeffs_dict = self.facemodel.split_coeff(output_coeff)


    def compute_losses(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        assert self.net_recog.training == False
        trans_m = self.trans_m
        if not self.opt.use_predef_M:
            trans_m = estimate_norm_torch(self.pred_lm, self.input_img.shape[-2])

        pred_feat = self.net_recog(self.pred_face, trans_m)
        gt_feat = self.net_recog(self.input_img, self.trans_m)
        self.loss_feat = self.opt.w_feat * self.compute_feat_loss(pred_feat, gt_feat)

        face_mask = self.pred_mask
        if self.opt.use_crop_face:
            face_mask, _, _ = self.renderer(self.pred_vertex, self.facemodel.front_face_buf)
        
        face_mask = face_mask.detach()
        self.loss_color = self.opt.w_color * self.comupte_color_loss(
            self.pred_face, self.input_img, self.atten_mask * face_mask)
        
        loss_reg, loss_gamma = self.compute_reg_loss(self.pred_coeffs_dict, self.opt)
        self.loss_reg = self.opt.w_reg * loss_reg
        self.loss_gamma = self.opt.w_gamma * loss_gamma

        self.loss_lm = self.opt.w_lm * self.compute_lm_loss(self.pred_lm, self.gt_lm)

        self.loss_reflc = self.opt.w_reflc * self.compute_reflc_loss(self.pred_tex, self.facemodel.skin_mask)

        self.loss_all = self.loss_feat + self.loss_color + self.loss_reg + self.loss_gamma \
                        + self.loss_lm + self.loss_reflc
            

    def optimize_parameters(self, isTrain=True):
        self.forward()               
        self.compute_losses()
        """Update network weights; it will be called in every training iteration."""
        if isTrain:
            self.optimizer.zero_grad()  
            self.loss_all.backward()         
            self.optimizer.step()        

    def compute_visuals(self):
        with torch.no_grad():
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
            
            if self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
                output_vis_numpy = util.draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
                output_vis_numpy = util.draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')
            
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw, output_vis_numpy), axis=-2)
            else:
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw), axis=-2)

            self.output_vis = torch.tensor(
                    output_vis_numpy / 255., dtype=torch.float32
                ).permute(0, 3, 1, 2).to(self.device)

    def save_mesh(self, name):

        recon_shape = self.pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1] # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = self.pred_color
        recon_color = recon_color.cpu().numpy()[0]
        tri = self.facemodel.face_buf.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8))
        mesh.export(name)

    def save_coeff(self,name):

        pred_coeffs = {key:self.pred_coeffs_dict[key].cpu().numpy() for key in self.pred_coeffs_dict}
        pred_lm = self.pred_lm.cpu().numpy()
        pred_lm = np.stack([pred_lm[:,:,0],self.input_img.shape[2]-1-pred_lm[:,:,1]],axis=2) # transfer to image coordinate
        pred_coeffs['lm68'] = pred_lm
        savemat(name,pred_coeffs)



