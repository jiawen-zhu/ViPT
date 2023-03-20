from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/got10k_lmdb'
    settings.got10k_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/itb'
    settings.lasot_extension_subset_path_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/lasot_lmdb'
    settings.lasot_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/lasot'
    settings.network_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/nfs'
    settings.otb_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/otb'
    settings.prj_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT'
    settings.result_plot_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/output/test/result_plots'
    settings.results_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/media/jiawen/Datasets/Codes/ViPT/ViPT/output'
    settings.segmentation_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/output/test/segmentation_results'
    settings.tc128_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/trackingnet'
    settings.uav_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/uav'
    settings.vot18_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/vot2018'
    settings.vot22_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/vot2022'
    settings.vot_path = '/media/jiawen/Datasets/Codes/ViPT/ViPT/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

