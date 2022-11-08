import logging
import copy
import os
import transforms3d as t3d
from .quaternion import qmul,qrot

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
import matplotlib.animation as animation
import numpy as np

import matplotlib.pyplot as plt




class Skeleton:
    """
       To calculate forward kinematics and save to ``*_skeleton.txt`` file
    Input:
        joints_parents: list of tuple ``(joints_name, parents_name)`` extracted from SkeletonInfo
        offsets: joints' offset to their parents
        init_rots(optional): joints' initial rotation to their parents
    """
    def __init__(self, joints_parents, offsets, init_rots=None):
        self.joints_parents = joints_parents
        self.offsets = offsets
        self.init_rots = init_rots

    def print_skel_tree(self, joint_name, layer_idx):
        if layer_idx >= 9:
            return
        print(' ' * layer_idx, '-', layer_idx, '-', joint_name)
        for cn_, pn_ in enumerate(self.joints_parents):
            if pn_ == joint_name:
                self.print_skel_tree(cn_, layer_idx+1)

    def trace_to_root(self, joint_name, parents_list):
        for cn_, pn_ in enumerate(self.joints_parents):
            if cn_ == joint_name:
                parents_list.append(pn_)
                self.trace_to_root(pn_, parents_list)
                break

    def forward_kinematics(self, rotations, root_positions=None):
        """
        :param rotations: dict, ``rotations[key] = array(n_frames, 4)``, local rotations
        :param root_positions:  (n_frames, 3)
        :return: world rotations, world positions
        """
        rotations_world = dict()
        positions_world = dict()

        first_name = list(rotations.keys())[0]

        # joints name, parents name
        for jn, pn in self.joints_parents:
            if jn not in rotations.keys() and self.init_rots:
                logging.warning('FK: !!! {} not in rotations keys, use initial rotation instead.'.format(jn))
                rotations[jn] = self.get_joint_init_rot(jn, rotations[first_name].shape[0])

            # only interpolation motions have this joint data and the motions in Dataset doesn't,
            # then use initial rotation
            if rotations[jn].shape[0] != rotations[first_name].shape[0]:
                logging.warning('FK: !!! Not all frames contain {} data, use initial rotation instead.'.format(jn))
                rotations[jn] = self.get_joint_init_rot(jn, rotations[first_name].shape[0])

            if pn == '-1':
                rotations_world[jn] = rotations[jn]
                if root_positions is None:
                    root_positions = np.zeros([rotations[jn].shape[0], 3])
                positions_world[jn] = root_positions
            else:
                rotations_world[jn] = qmul(rotations_world[pn], rotations[jn])
                offset = np.reshape(self.offsets[jn], [-1, 3])
                offset = np.repeat(offset, rotations_world[pn].shape[0], axis=0)
                positions_world[jn] = qrot(rotations_world[pn], offset) \
                                      + positions_world[pn]

        return rotations_world, positions_world

    def get_joint_init_rot(self, jn, n_frames):
        r = self.init_rots[jn]
        r = t3d.euler.euler2quat(r[2] * np.pi / 180, r[0] * np.pi / 180,
                                 r[1] * np.pi / 180, 'szxy')
        r = np.repeat(np.atleast_2d(r), n_frames, axis=0)
        return r

    @staticmethod
    def quat_w_last(q):
        """
        :param q: w, x, y ,z
        :return:  x, y, z, w
        """
        quat = copy.copy(q)
        quat[:3], quat[3] = quat[1:], quat[0]

        return quat


class SkeletonInfo:
    """
        Get joints names, their parents, their offsets to parents, and initial rotations
    Input
        skel_path: skeleton path, "*_skel.txt'
    """

    def __init__(self, skel_path):
        self._skel_path = skel_path

        # ``useful_joints.txt`` to exclude joints that not used in ``*_skeleton.txt`` file
        self._joints_path = os.path.join(os.path.dirname(skel_path), 'useful_joints.txt')

        self.joints, self.joints_parents, self.offsets, self.init_rots = self._get_skel_info()

    def _get_useful_joints(self):
        useful_joints = []

        try:
            with open(self._joints_path, "r") as fh:
                lines = fh.readlines()

            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                useful_joints.append(line)
        except FileNotFoundError:
            logging.warning('~~~Useful joints file path not found.~~~')

        return useful_joints

    def _get_skel_info(self):
        with open(self._skel_path, 'r', encoding='utf-8') as fh:
            skel_lines = fh.readlines()
        skel_lines = skel_lines[1:]

        joints_parents = dict()
        offsets = dict()
        rots = dict()
        for i, line in enumerate(skel_lines):
            parent_name, child_name = line.split(': ')[0].split(', ')

            pos_rot = line.split(': ')[1].strip()
            pos_rot = pos_rot.replace('(', '')
            pos_rot = pos_rot.replace(')', '')
            pos_rot = pos_rot.split(', ')
            pos_rot = [float(x) for x in pos_rot]
            offsets[child_name] = pos_rot[:3]
            rots[child_name] = pos_rot[3:]

            if i == 0:
                joints_parents[child_name] = '-1'
            else:
                joints_parents[child_name] = parent_name

        # get joints parents
        useful_joints = self._get_useful_joints()
        if not useful_joints:
            useful_joints = [jn for jn, pn in joints_parents]

        useful_joints_parents = []
        useful_offsets = dict()
        useful_rots = dict()

        for i, jn in enumerate(useful_joints):
            if jn not in joints_parents.keys():
                continue

            if i == 0:
                useful_joints_parents.append((jn, '-1'))
            else:
                useful_joints_parents.append((jn, joints_parents[jn]))

            useful_offsets[jn] = offsets[jn]
            useful_rots[jn] = rots[jn]

        return useful_joints, useful_joints_parents, useful_offsets, rots


class SkeletonData:
    """
        Get animation data of ``*_skeleton.txt``
    Input:
        skel_content: list of file lines in ``*_skeleton.txt``
    Return:
        rotations: local rotations, a dict like ``rotations[joints_name] = ndarray(n_frames, 4)``
        root_positions: ndarray(n_frames, 3)
    """
    def __init__(self, skel_content, data_version='v1'):
        """
        :param skel_content:
        :param data_version: skeleton data version, edit: 20191014
        """
        self.skel_content = skel_content
        if data_version == 'v1':
            self.rotations, self.root_positions = self._get_animation_data()
        elif data_version == 'v2':
            self.rotations, self.root_positions = self._get_animation_data_v2()
        else:
            raise NotImplementedError

    @staticmethod
    def quat_w_first(quat):
        """
        :param quat: x, y, z ,w
        :return: w, x, y, z
        """
        quat = quat[:, [3, 0, 1, 2]]

        return quat

    def _get_animation_data(self):
        rotations = dict()

        file_lines = self.skel_content
        for line in file_lines:
            line = line.strip()
            line_data = line.split(',')
            data_list = []
            for element in line_data:
                try:
                    value = float(element)
                    data_list.append(value)
                except ValueError:
                    element = element.strip()
                    if element == '':
                        continue
                    data_list = []
                    if element not in rotations:
                        rotations[element] = []

                    rotations[element].append(data_list)

        # process root position
        root_positions = None
        for k, v in rotations.items():
            if len(v[0]) == 7:
                root_name = k
                root_rotations = [x[3:] for x in rotations[root_name]]
                root_positions = np.asarray([x[:3] for x in rotations[root_name]])
                rotations[root_name] = root_rotations

        if root_positions is None:
            root_name = list(rotations.keys())[0]
            root_positions = np.zeros([rotations[root_name].shape[0], 3])

        for k, v in rotations.items():
            v = np.asarray(v)
            v = self.quat_w_first(v)
            rotations[k] = v

        return rotations, root_positions

    def _get_animation_data_v2(self):
        rotations = dict()

        file_lines = self.skel_content

        # get joints name
        joints_names = [jn for jn in file_lines[0].split(',') if jn not in ['\n', '']]
        root_positions = []
        for line in file_lines[1:]:
            line = line.strip()
            line_data = line.split(',')
            for i, jn in enumerate(joints_names):
                joint_data = [float(v) for v in line_data[i].split(' ')]
                if jn not in rotations.keys():
                    rotations[jn] = []

                if len(joint_data) == 7:
                    root_positions.append(joint_data[:3])
                    rotations[jn].append(joint_data[3:])
                else:
                    rotations[jn].append(joint_data)

        if not root_positions:
            root_name = list(rotations.keys())[0]
            root_positions = np.zeros([rotations[root_name].shape[0], 3])
        else:
            root_positions = np.asarray(root_positions)

        for k, v in rotations.items():
            v = np.asarray(v)
            v = self.quat_w_first(v)
            rotations[k] = v

        return rotations, root_positions

class SkelVisualization:
    """
        Visualize Skeleton Data
    Input:
        positions: a dict like ``positions[joints_name] = ndarray(nframes, 3)``
        joints_parents: list of tuple ``(joints_name, parents_name)``
        frame_rate: animation frames per second (FPS)
    """
    def __init__(self, positions, joints_parents, frame_rate=30):
        self.positions = positions
        self.joints_parents = joints_parents
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.lines = dict()
        self.frame_length = 1000 / frame_rate
        self.init_skel()
        self.set_lim()

    def init_skel(self, i=0):
        for jn, pn in self.joints_parents:
            if pn not in self.positions.keys():
                continue
            self.lines[jn] = self.ax.plot([self.positions[jn][i, 0], self.positions[pn][i, 0]],
                                          [self.positions[jn][i, 1], self.positions[pn][i, 1]],
                                          [self.positions[jn][i, 2], self.positions[pn][i, 2]],
                                          lw=4, zdir='y')

    def set_lim(self):
        all_pos = []
        for jn, pn in self.joints_parents:
            if pn not in self.positions.keys():
                continue
            all_pos.append(self.positions[jn])

        all_pos = np.asarray(all_pos)
        all_pos = np.reshape(all_pos, [-1, 3])
        min_lim = np.min(all_pos, axis=0)
        max_lim = np.max(all_pos, axis=0)
        max_max_lim = np.max(max_lim)
        min_min_lim = np.min(min_lim)

        self.ax.set_xlim3d(min_min_lim, max_max_lim)
        self.ax.set_ylim3d(min_min_lim, max_max_lim)
        self.ax.set_zlim3d(min_lim[1], max_lim[1])
        # self.ax.axis('equal')

    def update_skel(self, i):
        print(i)
        for jn, pn in self.joints_parents:
            if pn not in self.positions.keys():
                continue

            self.lines[jn][0].set_xdata([self.positions[jn][i, 0], self.positions[pn][i, 0]])
            self.lines[jn][0].set_ydata([self.positions[jn][i, 1], self.positions[pn][i, 1]])
            self.lines[jn][0].set_3d_properties(np.array([self.positions[jn][i, 2], self.positions[pn][i, 2]]), zdir='y')

    def animate(self):
        ani = animation.FuncAnimation(self.fig, self.update_skel,
                                      frames=self.positions[self.joints_parents[0][0]].shape[0],
                                      interval=self.frame_length, repeat=False, blit=False)
        plt.show()
        return ani



