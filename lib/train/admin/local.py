class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/media/jiawen/Datasets/Codes/ViPT/ViPT/pretrained_networks'
        self.got10k_val_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/got10k/val'
        self.lasot_lmdb_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/coco_lmdb'
        self.coco_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/coco'
        self.lasot_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/lasot'
        self.got10k_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/got10k/train'
        self.trackingnet_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/trackingnet'
        self.depthtrack_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/depthtrack/train'
        self.lasher_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/lasher/trainingset'
        self.visevent_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/visevent/train'
