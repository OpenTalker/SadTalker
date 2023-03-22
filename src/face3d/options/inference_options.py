from face3d.options.base_options import BaseOptions


class InferenceOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--dataset_mode', type=str, default=None, help='chooses how datasets are loaded. [None | flist]')

        parser.add_argument('--input_dir', type=str, help='the folder of the input files')
        parser.add_argument('--keypoint_dir', type=str, help='the folder of the keypoint files')
        parser.add_argument('--output_dir', type=str, default='mp4', help='the output dir to save the extracted coefficients')
        parser.add_argument('--save_split_files', action='store_true', help='save split files or not')
        parser.add_argument('--inference_batch_size', type=int, default=8)
        
        # Dropout and Batchnorm has different behavior during training and test.
        self.isTrain = False
        return parser
