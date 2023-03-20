import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.vot.vipt_class import run_vot_exp
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


run_vot_exp('vipt', 'deep_rgbd', vis=False, out_conf=True, channel_type='rgbd')
