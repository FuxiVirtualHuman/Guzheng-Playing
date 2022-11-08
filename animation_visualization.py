import argparse

from utils.skel_class import Skeleton,SkeletonInfo,SkeletonData,SkelVisualization

def parse_opts():
    parser = argparse.ArgumentParser(description='animation visualization')
    parser.add_argument('--skel_info_path', type=str,
                        default='./utils/ske_info/skel.txt',
                        help='path of skeleton information file')
    parser.add_argument('--animation_path', type=str,
                        default=r'',
                        help='path of animation file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_opts()
    # read animation data
    with open(opt.animation_path, 'r') as fh:
        skel_content = fh.readlines()

    skel_info = SkeletonInfo(opt.skel_info_path)
    skel_data = SkeletonData(skel_content, 'v1')
    skel = Skeleton(skel_info.joints_parents, skel_info.offsets, skel_info.init_rots)
    skel_data.root_positions = skel_data.root_positions - skel_data.root_positions
    rotations_world, positions_world = skel.forward_kinematics(skel_data.rotations, skel_data.root_positions)
    skel_vs = SkelVisualization(positions_world, skel_info.joints_parents)
    skel_vs.animate()