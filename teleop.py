import socket
import threading
import re
import math
import time
from SDK_LINUX import jkrc
from pyorbbecsdk import Config, Pipeline, OBSensorType, OBFormat, OBAlignMode, FrameSet
# from tools import *
import numpy as np
import signal
import queue  
import sys

import argparse
import cv2
import os
import asyncio
import websockets
from pickle import dumps, loads
from episode_writer import EpisodeWriter

message_stack = queue.LifoQueue()
main_vision_stack = queue.LifoQueue()
wrist_vision_stack = queue.LifoQueue()

# used to display image for display
render_frame_queue = queue.LifoQueue(maxsize=10)
send_main_view = False
send_wrist_view = False

PI = 3.1415926
COORD_BASE = 0
COORD_JOINT = 1
COORD_TOOL = 2
# 运动模式
ABS = 0
INCR = 1

rate = 800 # 控制机器人运动比例
vr_rot = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0]
])
vr_rot_inv = np.linalg.inv(vr_rot)

data_lock = threading.Lock()

latest_message = ""

# 退出标志
should_stop = False

# LGrip_Pressed = False
# RGrip_Pressed = False
left_trigger_pressed = False
right_trigger_pressed = False
accept_data = False

move_left_arm = False
move_right_arm = False
last_left_switch_time = 0
last_right_switch_time = 0
left_freeze_pose, right_freeze_pose = None, None

is_left_close = None
is_right_close = None

init_left_data = []
init_right_data = []

init_right_tcp_pos =[]
init_left_tcp_pos =[]

inv_left_init_rot = []
inv_right_init_rot = []
leftarm_init_rot = []
rightarm_init_rot = []

save = False


def clear_lifo_queue(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def start_server(host='0.0.0.0', port=8000):
    global should_stop, accept_data
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server is listening on {host}:{port}")
        while not should_stop:
            conn, addr = s.accept()
            with conn:
                # print(f"Connected by {addr}")
                buffer = ""  
                while not should_stop:
                    data = conn.recv(1024)
                    if not data:
                        # print("No data received, client disconnected.")
                        break
                    buffer += data.decode('utf-8')  
                    while '\n' in buffer:
                        index = buffer.index('\n') + 1
                        decoded_data = buffer[:index]  
                        buffer = buffer[index:] 

                        # 开始收数据
                        if not accept_data and ('LGrip=T' in decoded_data or 'RGrip=T' in decoded_data):
                            accept_data = True

                        # check message stack
                        if accept_data and decoded_data:
                            # print('buffer:',decoded_data, flush=True)
                            message_stack.put(decoded_data)

def main_camera_stream():
    global should_stop, accept_data, send_main_view
    """ 初始化主摄像头（奥比中光） """
    config = Config()
    pipeline = Pipeline()
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    # color_profile= profile_list.get_video_stream_profile(1280, 800, OBFormat.MJPG, 30)
    config.enable_stream(color_profile)
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    # depth_profile = profile_list.get_video_stream_profile(1280, 800, OBFormat.RLE, 30)
    config.enable_stream(depth_profile)
    print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
                                                color_profile.get_height(),
                                                color_profile.get_fps(),
                                                color_profile.get_format()))
    print("depth profile : {}x{}@{}_{}".format(depth_profile.get_width(),
                                                depth_profile.get_height(),
                                                depth_profile.get_fps(),
                                                depth_profile.get_format()))
    config.set_align_mode(OBAlignMode.HW_MODE)
    pipeline.enable_frame_sync()
    pipeline.start(config)
    # check main view image
    color_frame = None
    while color_frame is None:
        frames: FrameSet = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue
        color_data = np.asanyarray(color_frame.get_data())  # MJPG format
        color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)  # to get rgb image
        print(color_image.shape)
        cv2.imwrite(f'view_main_cam.jpg', color_image)
    
    while not should_stop:
        # 主视角捕获
        frames: FrameSet = pipeline.wait_for_frames(100)    # timeout in ms
        if frames is None:
            continue
        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            continue

        # 主视角读取
        color_data = np.asanyarray(color_frame.get_data())  # MJPG format
        # 放到共享内存
        if send_main_view:
            try:
                render_frame_queue.put(color_data, block=False)
            except:
                pass

        depth_data = depth_frame.get_data()  # RLE format
        
        if accept_data:
            main_vision_stack.put((color_data, depth_data))

    pipeline.stop()

def wrist_camera_stream(wrist_cam_indices):
    global should_stop, send_wrist_view
    """ 初始化腕部摄像头 """
    wrist_cams = []
    for i, device_index in enumerate(wrist_cam_indices):
        cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 获取视频高度
        fps = cap.get(cv2.CAP_PROP_FPS) # 获取帧率
        format = cap.get(cv2.CAP_PROP_FORMAT) # 获取格式
        print(f"wrist_{i} device_id:{device_index} width:{width} height:{height} fps:{fps} format:{format}")
        wrist_cams.append(cap)
        # check wrist view image
        ret, frame = cap.read() # 读取摄像头画面
        print(frame.shape)
        cv2.imwrite(f'view_wrist_{i}(cam_{device_index}).jpg', frame)
    
    while not should_stop:
        content = dict()
        # 腕部视角捕获
        valid = True
        for wrist_cam in wrist_cams:
            ret = wrist_cam.grab()
            if not ret:
                valid = False
        if not valid:
            continue
        # 腕部视角读取
        for i, wrist_cam in enumerate(wrist_cams):
            _, frame = wrist_cam.retrieve()
            content[f"rgb_wrist_{i}"] = frame

        if send_wrist_view:
            try:
                render_frame_queue.put(content, block=False)
            except:
                pass
        
        # 放到共享内存
        if accept_data:
            wrist_vision_stack.put(content)


def process_data(decoded_data):
    global should_stop,accept_data,left_trigger_pressed,right_trigger_pressed
    global move_left_arm, move_right_arm, last_left_switch_time, last_right_switch_time
    if (time.time() - last_left_switch_time) > 5 and 'PauseL=T' in decoded_data:
        move_left_arm = not move_left_arm
        last_left_switch_time = time.time()
        print('PauseL=T')
    if (time.time() - last_right_switch_time) > 5 and 'PauseR=T' in decoded_data:
        move_right_arm = not move_right_arm
        last_right_switch_time = time.time()
        print('PauseR=T')
    if 'EXIT=T'in decoded_data:
        accept_data = False
        should_stop = True
        print('EXIT=T')
    # 夹
    if move_left_arm:
        if 'LTrig=T' in decoded_data:
            left_trigger_pressed = True
        if 'LTrig=F' in decoded_data:
            left_trigger_pressed = False
    if move_right_arm:
        if 'RTrig=T' in decoded_data:
            right_trigger_pressed = True
        if 'RTrig=F' in decoded_data:
            right_trigger_pressed = False
    if accept_data:
        # print("----------------------------accept_data") 
        robot_data = robot_control(decoded_data)
        return robot_data
    else:
        # print("waitting")
        return None


def euler_to_rotation_matrix(rx, ry, rz):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    R = np.dot(Ry, np.dot(Rx, Rz))
    return R


def robot_control(decoded_data):
    global should_stop, save, is_left_close, is_right_close
    global rate, vr_rot, vr_rot_inv, inv_left_init_rot, inv_right_init_rot
    global leftarm_init_rot,rightarm_init_rot,init_left_data,init_right_data,init_right_tcp_pos,init_left_tcp_pos
    global left_trigger_pressed,right_trigger_pressed, left_freeze_pose, right_freeze_pose

    with data_lock:
        start_time = time.time()  
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', decoded_data)
        if len(numbers) < 12:
            # print("Not enough data to process.")
            return None
        numbers = [float(num) for num in numbers]
        
        left_pos = [numbers[0] * rate, numbers[1] *rate, numbers[2] * rate]
        right_pos = [numbers[6] * rate, numbers[7] * rate, numbers[8] * rate]
        left_rot_rad = [math.radians(angle) for angle in numbers[3:6]]
        right_rot_rad = [math.radians(angle) for angle in numbers[9:12]]
        left_data = left_pos + left_rot_rad
        right_data = right_pos + right_rot_rad
        if not init_left_data:
            init_left_data = left_data[:]
            left_init_rot = euler_to_rotation_matrix(-init_left_data[3], -init_left_data[4], init_left_data[5])  #yxz
            inv_left_init_rot = np.linalg.inv(left_init_rot)
        if not init_right_data:
            init_right_data = right_data[:]
            right_init_rot = euler_to_rotation_matrix(-init_right_data[3], -init_right_data[4], init_right_data[5])
            inv_right_init_rot = np.linalg.inv(right_init_rot)

        left_diff = [a - b for a, b in zip(left_data, init_left_data)]
        right_diff = [a - b for a, b in zip(right_data, init_right_data)]

        left_rot = euler_to_rotation_matrix(-left_data[3], -left_data[4], left_data[5])
        right_rot = euler_to_rotation_matrix(-right_data[3], -right_data[4], right_data[5])

        left_rotvr_diff = np.dot(left_rot, inv_left_init_rot)
        right_rotvr_diff = np.dot(right_rot, inv_right_init_rot)
        
        left_diff_base= np.dot(vr_rot,np.dot(left_rotvr_diff,vr_rot_inv))
        right_diff_base= np.dot(vr_rot,np.dot(right_rotvr_diff,vr_rot_inv))


        _, left_robot_status = left_arm.get_robot_status()
        # for i, v in enumerate(left_robot_status):
        #     print(i, v)
        left_ref_pos = left_robot_status[19]
        left_tcp_pos = left_robot_status[18]
        if not init_left_tcp_pos:
            init_left_tcp_pos = left_tcp_pos
            left_rpy = [init_left_tcp_pos[3], init_left_tcp_pos[4], init_left_tcp_pos[5]]
            leftarm_init_rot = left_arm.rpy_to_rot_matrix(left_rpy)

        leftarm_finalrot = np.dot(left_diff_base,leftarm_init_rot[1]) 
        left_rot_diff = left_arm.rot_matrix_to_rpy(leftarm_finalrot)
        left_next_pos = list(init_left_tcp_pos)
        left_next_pos[0] = init_left_tcp_pos[0] + left_diff[2]
        left_next_pos[1] = init_left_tcp_pos[1] - left_diff[0]
        left_next_pos[2] = init_left_tcp_pos[2] + left_diff[1]

        left_next_pos[3] = left_rot_diff[1][0]
        left_next_pos[4] = left_rot_diff[1][1]
        left_next_pos[5] = left_rot_diff[1][2]
        # print('重映射用时:', time.time() - start_time)

        if not move_left_arm:
            left_next_pos = [(a*4 + b*1)/ (4 + 1) for (a, b) in zip(left_freeze_pose, left_next_pos)]
            print("Freeze Left")
        else:
            left_freeze_pose = left_next_pos

        # start_time = time.time()
        left_arm_ret = left_arm.kine_inverse7(left_ref_pos,left_next_pos)
        # print('逆解用时(left):', time.time() - start_time)

        _, right_robot_status = right_arm.get_robot_status()
        right_ref_pos = right_robot_status[19]
        right_tcp_pos = right_robot_status[18]
        if not init_right_tcp_pos:
            init_right_tcp_pos = right_tcp_pos
            right_rpy = [init_right_tcp_pos[3], init_right_tcp_pos[4], init_right_tcp_pos[5]]
            rightarm_init_rot = right_arm.rpy_to_rot_matrix(right_rpy)

        rightarm_finalrot = np.dot(right_diff_base,rightarm_init_rot[1]) 
        right_rot_diff = right_arm.rot_matrix_to_rpy(rightarm_finalrot)

        right_next_pos = list(init_right_tcp_pos)
        right_next_pos[0] = init_right_tcp_pos[0] + right_diff[2]
        right_next_pos[1] = init_right_tcp_pos[1] - right_diff[0]
        right_next_pos[2] = init_right_tcp_pos[2] + right_diff[1]

        right_next_pos[3] = right_rot_diff[1][0]
        right_next_pos[4] = right_rot_diff[1][1]
        right_next_pos[5] = right_rot_diff[1][2]

        if not move_right_arm:
            right_next_pos = [(a*4 + b*1)/ (4 + 1) for (a, b) in zip(right_freeze_pose, right_next_pos)]
            print("Freeze Right")
        else:
            right_freeze_pose = right_next_pos

        # start_time = time.time()
        right_arm_ret = right_arm.kine_inverse7(right_ref_pos,right_next_pos)
        # print('逆解用时(right):', time.time() - start_time)

        start_time = time.time()
        if left_arm_ret[0] == 0:
            ## count+=1               
            left_ret = left_arm.servo_j7(left_arm_ret[1],ABS)
            # left_ret = left_arm.joint_move7(left_arm_ret[1], move_mode=ABS, is_block=1, speed=150)
            if left_ret[0] ==0:
                l_joint_abs_act = left_arm_ret[1]
                l_eef_abs_act = left_next_pos
            elif left_ret[0] in [-1, -7]:
                print(left_ret)
                print('左臂退出')
                should_stop = True
                save = False
                return None
            else:
                print('left arm 移动失败', left_ret)
                left_arm.collision_recover()
                left_arm.enable_robot()
                left_arm.servo_move_enable(1)
                l_joint_abs_act = left_arm.get_joint_position()[1]
                l_eef_abs_act = left_arm.get_tcp_position()[1]
        else:
            print('left arm 逆解失败', left_arm_ret)
            l_joint_abs_act = left_ref_pos
            l_eef_abs_act = left_tcp_pos
        
        if right_arm_ret[0] == 0:
            right_ret = right_arm.servo_j7(right_arm_ret[1],ABS)
            # right_ret = right_arm.joint_move7(right_arm_ret[1], move_mode=ABS, is_block=1, speed=150)
            if right_ret[0] ==0:
                r_joint_abs_act = right_arm_ret[1]
                r_eef_abs_act = right_next_pos
            elif right_ret[0] in [-1, -7]:
                print(right_ret)
                print('右臂退出')
                should_stop = True
                save = False
                return None
            else:
                # print('right arm 移动失败', right_ret)
                right_arm.collision_recover()
                right_arm.enable_robot()
                right_arm.servo_move_enable(1)
                r_joint_abs_act = right_arm.get_joint_position()[1]
                r_eef_abs_act = right_arm.get_tcp_position()[1]
        else:
            print('right arm 逆解失败', right_arm_ret)
            r_joint_abs_act = right_ref_pos
            r_eef_abs_act = right_tcp_pos
        # print('运动伺服用时(left+right):', time.time() - start_time)

        start_time = time.time()
        if open_width > 0 and is_left_close != left_trigger_pressed:
            left_arm.set_analog_output(iotype = 2,index = 3,value = close_width if left_trigger_pressed else open_width)
        is_left_close = left_trigger_pressed
        if open_width > 0 and is_right_close != right_trigger_pressed:
            right_arm.set_analog_output(iotype = 2,index = 3,value = close_width if right_trigger_pressed else open_width)
        is_right_close = right_trigger_pressed
        # print('夹爪控制用时(left+right):', time.time() - start_time)
        
        states = {
            'joint_pos_l': left_ref_pos,
            'joint_pos_r': right_ref_pos,
            'eef_pose_l': left_tcp_pos,
            'eef_pose_r': right_tcp_pos,
            'joint_status_l': left_robot_status[20][5],
            'joint_status_r': right_robot_status[20][5],
            'eef_FT_l': left_robot_status[21][5:],
            'eef_FT_r': right_robot_status[21][5:],
            'gripper_state_l': left_robot_status[17],
            'gripper_state_r': right_robot_status[17],
        }
        actions = {
            'act_joint_pos_l': l_joint_abs_act,
            'act_joint_pos_r': r_joint_abs_act,
            'act_eef_pose_l': l_eef_abs_act,
            'act_eef_pose_r': r_eef_abs_act,
            'act_grip_l': left_trigger_pressed,
            'act_grip_r': right_trigger_pressed,
        }
        return {'states': states, 'actions': actions}


def run_server():
    # 接收手柄消息的ip和端口号
    server_thread = threading.Thread(target=start_server, args=('192.168.5.100', 8000))  
    server_thread.start()
    return server_thread

def run_main_camera_server():
    # 推送主摄像头数据
    main_camera_server_thread = threading.Thread(target=main_camera_stream)
    main_camera_server_thread.start()
    return main_camera_server_thread

def run_wrist_camera_server(wrist_cam_indices):
    # 推送腕部摄像头数据
    wrist_camera_server_thread = threading.Thread(target=wrist_camera_stream, kwargs={"wrist_cam_indices": wrist_cam_indices})
    wrist_camera_server_thread.start()
    return wrist_camera_server_thread

async def streaming_main_view(websocket, path):
    global should_stop, send_main_view
    send_main_view = True
    while not should_stop:
        try:
            await websocket.send(render_frame_queue.get(block=False).tobytes())
        except:
            continue
    loop = asyncio.get_event_loop()
    loop.stop()

async def streaming_wrist_view(websocket, path):
    global should_stop, send_wrist_view
    send_wrist_view = True
    while not should_stop:
        try:
            content = render_frame_queue.get(block=False)
            comb_img = np.hstack([content[f"rgb_wrist_0"], content[f"rgb_wrist_1"]])
            success, jpg_buffer = cv2.imencode('.jpg', comb_img)
            if not success:
                continue
            await websocket.send(jpg_buffer.tobytes())
        except:
            continue
    loop = asyncio.get_event_loop()
    loop.stop()
    

def start_asyncio_server(streaming_func):
    try:
        loop = asyncio.new_event_loop()  # 创建新事件循环
        asyncio.set_event_loop(loop)  # 设置当前线程的事件循环
        loop.run_until_complete(  # 在新事件循环中运行
            websockets.serve(streaming_func, "192.168.5.100", 8081)  # send camera data in this address
        )
        print(f"start websockets at 8081")
        loop.run_forever()
    except:
        return

# For Debug
from flask import Flask, send_file
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
@app.route('/')
def web_page():
    return send_file('show_cam_data.html')
def run_flask_app():
    app.run(host='0.0.0.0', port=8080, use_reloader=False, threaded=True)

def main_loop():
    global should_stop
    global save
    if save:
        recorder = EpisodeWriter(
            data_dir=data_dir, task=task, close_width=close_width, open_width=open_width, frequency=frequency)
        recorder.create_episode()
    cnt = 0
    while not should_stop:
        if not message_stack.empty():
            if cnt == 0:
                previous_time = time.time()
            
            # start_time = time.time()
            decoded_data = message_stack.get()
            # print('捕获VR信息用时:', time.time() - start_time)
            robot_data = process_data(decoded_data)
            # robot_data = {"states": None, "actions": None}
            wrist_vision_data = wrist_vision_stack.get()
            main_vision_data = main_vision_stack.get()
            if robot_data is None:
                continue
            if save:
                recorder.add_item(main_vision_data=main_vision_data, wrist_vision_data=wrist_vision_data, robot_data=robot_data)
            cnt += 1
            current_time = time.time()
            time_elapsed = current_time - previous_time
            sleep_time = (1 / frequency) - time_elapsed
            print(time_elapsed, flush=True)
            if sleep_time > 0:
                time.sleep(sleep_time)
            previous_time = time.time()
        elif accept_data:
            print("Waiting for data...", flush=True)
            continue
    if save:
        if cnt > 400:
            print(f"Discard the episode that is too long ({cnt} steps)")
            return
        clear_lifo_queue(message_stack)
        clear_lifo_queue(main_vision_stack)
        clear_lifo_queue(wrist_vision_stack)
        recorder.save_episode()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='debug', help='Directory for saving the data')
    parser.add_argument('--task', type=str, default='play')
    parser.add_argument('--close_width', type=int, default=0, help='Gripper closed width')
    parser.add_argument('--open_width', type=int, default=700, help='Gripper open width')
    parser.add_argument('--left_wrist_cam_id', type=int, default=8, help='Opencv device index of left wrist camera')
    parser.add_argument('--right_wrist_cam_id', type=int, default=6, help='Opencv device index of right wrist camera')
    parser.add_argument('--frequency', type=int, default=10, help='Frequency of teleoperation')
    parser.add_argument('--rate', type=int, default=800, help='控制机器人运动比例')
    parser.add_argument('--save', action='store_true', default=False, help='Save data or not')
    parser.add_argument('--display', action='store_true', default=False, help='Display on browser')
    parser.add_argument('--streaming', action='store_true', default=False, help='Streaming onto VR')
    args = parser.parse_args()

    data_dir = args.data_dir
    task = args.task
    close_width = args.close_width
    open_width = args.open_width
    frequency = args.frequency
    rate = args.rate
    save = args.save
    display = args.display

    move_left_arm = True
    move_right_arm = True # False

    is_left_close = (open_width == 0)
    is_right_close = (open_width == 0)

    # left arm
    left_arm = jkrc.RC("192.168.5.102")    # robot ip
    left_ret = left_arm.login()
    # left_arm.disable_robot()
    # if left_arm.is_in_collision()[1] == 1:
    #     left_arm.collision_recover()
    # left_arm.power_off()
    # left_arm.power_on()  
    left_ret = left_arm.enable_robot()
    if left_ret[0] != 0:
        print("Left arm failed to start.")
        exit(0)
    left_arm.set_analog_output(iotype = 2,index = 3,value = open_width)
    left_arm.servo_move_use_joint_LPF(0.3)  
    left_arm.servo_move_enable(1)
    print("Left arm is ready.")
    # ret = left_arm.set_full_dh_flag(1)
    # if ret[0] != 0:
    #     print("Fail to set full DH for left arm.", ret)
    #     exit(0)
    
    # right arm
    right_arm = jkrc.RC("192.168.5.200")    # robot ip 
    right_ret = right_arm.login()
    # right_arm.disable_robot()
    # if right_arm.is_in_collision()[1] == 1:
    #     right_arm.collision_recover()
    # right_arm.power_off()
    # right_arm.power_on()
    right_ret = right_arm.enable_robot()
    if right_ret[0] != 0:
        print("Right arm failed to start.")
        exit(0)
    right_arm.set_analog_output(iotype = 2,index = 3,value = open_width)
    right_arm.servo_move_use_joint_LPF(0.3) 
    right_arm.servo_move_enable(1)
    print("Right arm is ready.")
    # ret = right_arm.set_full_dh_flag(1)
    # if ret[0] != 0:
    #     print("Fail to set full DH for right arm.", ret)
    #     exit(0)

    # 启动服务器线程
    server_thread = run_server()
    main_camera_server_thread = run_main_camera_server()
    wrist_camera_server_thread = run_wrist_camera_server(wrist_cam_indices=[args.left_wrist_cam_id, args.right_wrist_cam_id])
    main_thread = threading.Thread(target=main_loop)
    # 启动主循环线程
    main_thread.start()

    if display or args.streaming:
        # 启动异步 WebSocket 服务器用于直播主视角（VR显示或浏览器显示）
        asyncio_thread = threading.Thread(target=start_asyncio_server, kwargs={"streaming_func": streaming_main_view})
        asyncio_thread.start()
        # 等待异步服务器线程启动
        asyncio_thread.join(timeout=1.0)  # 可选，根据需要调整

    if display:
        # 启动 Flask 应用查看主摄像头图像
        flask_thread = threading.Thread(target=run_flask_app)
        flask_thread.start()

    # 等待服务器线程和主循环线程结束
    server_thread.join()
    main_camera_server_thread.join()
    wrist_camera_server_thread.join()
    main_thread.join()

    if should_stop:
        exit(0)
