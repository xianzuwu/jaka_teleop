import os
import cv2
import json
import datetime
import numpy as np
import time

from pickle import loads
from collections import defaultdict
from moviepy.editor import ImageSequenceClip

class EpisodeWriter(object):
    def __init__(self, data_dir, task, close_width=50, open_width=500, frequency=30, main_img_size=[1280, 720], wrist_img_size=[640, 480],
                 main_img_is_mjpg=True, version='1.0.0', date=None, operator=None):
        """
        main_img_size: [width, height]
        """
        print("==> EpisodeWriter initializing...\n")
        self.data_dir = data_dir
        self.task = task
        self.close_width = close_width
        self.open_width = open_width
        self.frequency = frequency
        self.main_img_size = main_img_size
        self.wrist_img_size = wrist_img_size
        self.main_img_is_mjpg = main_img_is_mjpg
        
        self.data = {}
        self.episode_data = []
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.data_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.data_dir) if 'episode_' in episode_dir]
            if len(episode_dirs) > 0:
                episode_last = sorted(episode_dirs)[-1]
                self.episode_id = int(episode_last.split('_')[-1])
                print(f"==> Data directory already exists ({self.episode_id+1} episodes).\n")
            else:
                print(f"==> An empty data directory exists.\n")
        else:
            os.makedirs(self.data_dir)
            print(f"==> Data directory does not exist, now create one.\n")
        
        self.version = version
        self.date = date
        self.operator = operator
        print("==> EpisodeWriter initialized successfully.\n")

    def create_episode(self):
        """
        Create a new episode, each episode needs to specify the episode_id.
        """

        self.item_id = -1
        self.episode_data = []
        self.episode_id = self.episode_id + 1
        
        self.episode_dir = os.path.join(self.data_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.video_dir = os.path.join(self.episode_dir, 'videos')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        
    def add_item(self, main_vision_data, wrist_vision_data, robot_data):
        self.episode_data.append((main_vision_data, wrist_vision_data, robot_data))

    def save_episode(self):
        """
            with open("./hmm.json",'r',encoding='utf-8') as json_file:
                model=json.load(json_file)
        """
        self.data['info'] = {
            "version": "1.0.0" if self.version is None else self.version, 
            "datetime": datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") if self.date is None else self.date, 
            "operator": "someone" if self.operator is None else self.operator,
            "main_cam_info": {"width":self.main_img_size[0], "height":self.main_img_size[1]},
            "wrist_cam_info": {"width":self.wrist_img_size[0], "height":self.wrist_img_size[1]},
            "frequency": self.frequency,
            "close_width": self.close_width,
            "open_width": self.open_width,
            "episode_dir": self.episode_dir,
            "episode_id": self.episode_id,
            "task": self.task,
            "total_steps": len(self.episode_data),
        }
        episode_items = []
        video_dict = defaultdict(list)
        for (main_vision_data, wrist_vision_data, robot_data) in self.episode_data:
            self.item_id += 1
            item_data = {
                'idx': self.item_id,
                'colors': {'rgb_main': None, 'rgb_wrist_0': None, 'rgb_wrist_1': None},
                'depths': {'depth_main': None},
            }
            
            combine_img = np.zeros((self.main_img_size[1]+self.wrist_img_size[1], self.wrist_img_size[0]*2, 3), dtype=np.uint8)
            main_x_axis_shif = (self.wrist_img_size[0]*2 - self.main_img_size[0]) // 2

            # 主视觉
            rgb_main_data, depth_main_data = main_vision_data
            # rgb_main
            rgb_data = cv2.imdecode(rgb_main_data, cv2.IMREAD_COLOR) if self.main_img_is_mjpg else rgb_main_data
            combine_img[:self.main_img_size[1], main_x_axis_shif:main_x_axis_shif+self.main_img_size[0]] = rgb_data
            save_dir = self.color_dir
            save_name = f'{str(self.item_id).zfill(6)}_rgb_main.jpg'
            cv2.imwrite(os.path.join(save_dir, save_name), rgb_data)
            item_data['colors']['rgb_main'] = os.path.join('colors', save_name)
            # depth_main
            depth_data = np.frombuffer(depth_main_data, dtype=np.uint16)
            save_dir = self.depth_dir
            # save_name = f'{str(self.item_id).zfill(6)}_depth_main.npy'
            # np.save(os.path.join(save_dir, save_name), depth_data)
            save_name = f'{str(self.item_id).zfill(6)}_depth_main.npz'
            np.savez_compressed(os.path.join(save_dir, save_name), array=depth_data)
            item_data['depths']['depth_main'] = os.path.join('depths', save_name)

            # 腕部视觉
            for (k, v) in wrist_vision_data.items():
                save_dir = self.color_dir
                save_name = f'{str(self.item_id).zfill(6)}_{k}.jpg'
                rgb_data = v
                wrist_id = int(k.split('_')[-1])
                x_axis_shif = wrist_id * self.wrist_img_size[0]
                combine_img[self.main_img_size[1]:, x_axis_shif:x_axis_shif+self.wrist_img_size[0]] = rgb_data
                cv2.imwrite(os.path.join(save_dir, save_name), rgb_data)
                item_data['colors'][k] = os.path.join('colors', save_name)
            video_dict['combine'].append(combine_img)

            states = dict()
            for k, v in robot_data['states'].items():
                v_ = v
                if "joint_status" in k:
                    velocity = [x[-2] for x in v]
                    torque = [x[-1] for x in v]
                    v_ = {"velocity": velocity, "torque": torque}
                if "eef_FT" in k:
                    raw_value = v[1]
                    actual_contact_FT = v[0]
                    actual_contact_FT_ = v[2]
                    v_ = {"raw_value": raw_value, "actual_contact_FT": actual_contact_FT}
                if "gripper_state" in k:
                    ext_input = list(v[2])
                    state_flag = ext_input[1]
                    position = ext_input[2]
                    motor_current = ext_input[4]
                    # covert the gripper motor current value (the 5th register value of external IO input) from uint16 to signed int16
                    if motor_current & 0x8000:
                        motor_current = motor_current - 0x10000
                    ext_output = list(v[3])
                    control_force = ext_output[1]
                    control_position = ext_output[3]
                    v_ = {"state_flag": state_flag, "position": position, "motor_current": motor_current, "control_force": control_force, "control_position": control_position}
                states[k] = v_
            item_data['states'] = states

            item_data['actions'] = robot_data['actions']
            episode_items.append(item_data)
        self.data['data'] = episode_items
        
        for k, v in video_dict.items():
            images = [img[:,:,::-1] for img in v]
            video = ImageSequenceClip(images, fps=self.frequency)
            video.write_videofile(os.path.join(self.video_dir, k)+'.mp4', fps=self.frequency, logger=None)
        with open(self.json_path,'w',encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(self.data, indent=2, ensure_ascii=False))
        