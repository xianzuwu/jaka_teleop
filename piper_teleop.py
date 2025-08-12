import socket
import threading
import re
import math
import time
from piper_sdk import *
# from tools import *
import numpy as np
import signal
import queue  
import sys

import argparse
import cv2
import os

from pickle import dumps, loads
from episode_writer import EpisodeWriter

message_stack = queue.LifoQueue()
main_vision_stack = queue.LifoQueue()
wrist_vision_stack = queue.LifoQueue()

PI = 3.1415926
# COORD_BASE = 0
# COORD_JOINT = 1
# COORD_TOOL = 2
# # 运动模式
# ABS = 0
# INCR = 1

# rate = 800 # 控制机器人运动比例
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
right_trigger_pressed = False
accept_data = False

move_right_arm = False

last_right_switch_time = 0
right_freeze_pose = None

is_right_close = None

init_right_data = []

init_right_tcp_pos =[]



inv_right_init_rot = []
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
                        if not accept_data and ('RGrip=T' in decoded_data):
                            accept_data = True

                        # check message stack
                        if accept_data and decoded_data:
                            # print('buffer:',decoded_data, flush=True)
                            message_stack.put(decoded_data)


def wrist_camera_stream(device_index):
    global should_stop, accept_data
    """ 初始化腕部摄像头 """
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 获取视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 获取视频高度
    fps = cap.get(cv2.CAP_PROP_FPS) # 获取帧率
    format = cap.get(cv2.CAP_PROP_FORMAT) # 获取格式
    print(f"[WristCam] ID={device_index} width={width} height={height} fps={fps} format={format}")
    
    ret, frame = cap.read()
    print(frame.shape)
    if ret:
        cv2.imwrite(f'view_wrist(cam_{device_index}).jpg', frame)
    
    try:
        while not should_stop:
            content = dict()
            # 腕部视角捕获
            valid = True
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 腕部视角读取
            content["rgb_wrist_0"] = frame
            
            # 放到共享内存
            if accept_data:
                wrist_vision_stack.put(content)
    finally:
        cap.release()
        print("[WristCam] released.")


def process_data(decoded_data):
    global should_stop,accept_data,right_trigger_pressed
    global move_right_arm, last_right_switch_time

    if (time.time() - last_right_switch_time) > 5 and 'PauseR=T' in decoded_data:
        move_right_arm = not move_right_arm
        last_right_switch_time = time.time()
        print('PauseR=T')
    if 'EXIT=T'in decoded_data:
        accept_data = False
        should_stop = True
        print('EXIT=T')
    # 夹

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
    # rx,ry,rz: roll, pitch, yaw（弧度）
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx   # ← ZYX


def robot_control(decoded_data):
    global should_stop, save, is_right_close
    global rate, vr_rot, vr_rot_inv, inv_right_init_rot
    global rightarm_init_rot, init_right_data, init_right_tcp_pos
    global right_trigger_pressed, right_freeze_pose, move_right_arm
    global piper

    # with data_lock:
    #     start_time = time.time()  
    #     numbers = re.findall(r'[-+]?\d*\.\d+|\d+', decoded_data)
    #     if len(numbers) < 6:
    #         # print("Not enough data to process.")
    #         return None
    #     numbers = [float(num) for num in numbers]
        
    #     right_pos = [numbers[6] * rate, numbers[7] * rate, numbers[8] * rate]
    #     right_rot_rad = [math.radians(angle) for angle in numbers[9:12]]
    #     right_data = right_pos + right_rot_rad

    #     if not init_right_data:
    #         init_right_data = right_data[:]
    #         right_init_rot = euler_to_rotation_matrix(-init_right_data[3], -init_right_data[4], init_right_data[5])
    #         inv_right_init_rot = np.linalg.inv(right_init_rot)

    #     # VR 差分(位置/姿态)
    #     right_diff = [a - b for a, b in zip(right_data, init_right_data)]
    #     right_rot = euler_to_rotation_matrix(-right_data[3], -right_data[4], right_data[5])
    #     right_rotvr_diff = np.dot(right_rot, inv_right_init_rot)
    #     right_diff_base= np.dot(vr_rot,np.dot(right_rotvr_diff,vr_rot_inv))

    #     _, right_robot_status = right_arm.get_robot_status()
    #     right_ref_pos = right_robot_status[19]
    #     right_tcp_pos = right_robot_status[18]
    #     if not init_right_tcp_pos:
    #         init_right_tcp_pos = right_tcp_pos
    #         right_rpy = [init_right_tcp_pos[3], init_right_tcp_pos[4], init_right_tcp_pos[5]]
    #         rightarm_init_rot = right_arm.rpy_to_rot_matrix(right_rpy)

    #     rightarm_finalrot = np.dot(right_diff_base,rightarm_init_rot[1]) 
    #     right_rot_diff = right_arm.rot_matrix_to_rpy(rightarm_finalrot)

    #     right_next_pos = list(init_right_tcp_pos)
    #     right_next_pos[0] = init_right_tcp_pos[0] + right_diff[2]
    #     right_next_pos[1] = init_right_tcp_pos[1] - right_diff[0]
    #     right_next_pos[2] = init_right_tcp_pos[2] + right_diff[1]

    #     right_next_pos[3] = right_rot_diff[1][0]
    #     right_next_pos[4] = right_rot_diff[1][1]
    #     right_next_pos[5] = right_rot_diff[1][2]

    #     if not move_right_arm:
    #         right_next_pos = [(a*4 + b*1)/ (4 + 1) for (a, b) in zip(right_freeze_pose, right_next_pos)]
    #         print("Freeze Right")
    #     else:
    #         right_freeze_pose = right_next_pos

    #     # start_time = time.time()
    #     right_arm_ret = right_arm.kine_inverse7(right_ref_pos,right_next_pos)
    #     # print('逆解用时(right):', time.time() - start_time)

    #     start_time = time.time()
        
    #     if right_arm_ret[0] == 0:
    #         right_ret = right_arm.servo_j7(right_arm_ret[1],ABS)
    #         # right_ret = right_arm.joint_move7(right_arm_ret[1], move_mode=ABS, is_block=1, speed=150)
    #         if right_ret[0] ==0:
    #             r_joint_abs_act = right_arm_ret[1]
    #             r_eef_abs_act = right_next_pos
    #         elif right_ret[0] in [-1, -7]:
    #             print(right_ret)
    #             print('右臂退出')
    #             should_stop = True
    #             save = False
    #             return None
    #         else:
    #             # print('right arm 移动失败', right_ret)
    #             right_arm.collision_recover()
    #             right_arm.enable_robot()
    #             right_arm.servo_move_enable(1)
    #             r_joint_abs_act = right_arm.get_joint_position()[1]
    #             r_eef_abs_act = right_arm.get_tcp_position()[1]
    #     else:
    #         print('right arm 逆解失败', right_arm_ret)
    #         r_joint_abs_act = right_ref_pos
    #         r_eef_abs_act = right_tcp_pos
    #     # print('运动伺服用时(left+right):', time.time() - start_time)

    #     start_time = time.time()
    #     if open_width > 0 and is_right_close != right_trigger_pressed:
    #         right_arm.set_analog_output(iotype = 2,index = 3,value = close_width if right_trigger_pressed else open_width)
    #     is_right_close = right_trigger_pressed
    #     # print('夹爪控制用时(left+right):', time.time() - start_time)
        
        states = {
            'joint_pos_r': right_ref_pos,
            'eef_pose_r': right_tcp_pos,
            'joint_status_r': right_robot_status[20][5],
            'eef_FT_r': right_robot_status[21][5:],
            'gripper_state_r': right_robot_status[17],
        }
        actions = {
            'act_joint_pos_r': r_joint_abs_act,
            'act_eef_pose_r': r_eef_abs_act,
            'act_grip_r': right_trigger_pressed,
        }
        return {'states': states, 'actions': actions}


def run_server():
    # 接收手柄消息的ip和端口号
    server_thread = threading.Thread(target=start_server, args=('192.168.5.100', 8000))  
    server_thread.start()
    return server_thread


def run_wrist_camera_server(wrist_cam_id):
    # 推送腕部摄像头数据
    wrist_camera_server_thread = threading.Thread(target=wrist_camera_stream, args=(wrist_cam_id,), daemon=True)
    wrist_camera_server_thread.start()
    return wrist_camera_server_thread

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
            main_vision_data = None
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
    parser.add_argument('--right_wrist_cam_id', type=int, default=6, help='Opencv device index of right wrist camera')
    parser.add_argument('--frequency', type=int, default=10, help='Frequency of teleoperation')
    parser.add_argument('--rate', type=int, default=800, help='控制机器人运动比例')
    parser.add_argument('--save', action='store_true', default=False, help='Save data or not')

    args = parser.parse_args()

    data_dir = args.data_dir
    task = args.task
    close_width = args.close_width
    open_width = args.open_width
    frequency = args.frequency
    rate = args.rate
    save = args.save

    move_right_arm = True # False

    is_right_close = (open_width == 0)


    # right arm (Piper) ✅
    piper = C_PiperInterface_V2('can0')          # 实例化Piper Arm
    piper.ConnectPort()                         # 连接/登录
    while( not piper.EnablePiper()):
        time.sleep(0.01)
    print("Piper arm is ready.")
    
    piper.set_servo_mode("cartesian")         # 进入笛卡尔伺服模式（如有）
    piper.set_lpf(0.3)                        # 如果有低通参数接口
    if open_width > 0:
        piper.gripper_open(width=open_width)  # 初始张开夹爪

    # 启动服务器线程
    server_thread = run_server()

    wrist_camera_server_thread = run_wrist_camera_server(wrist_cam_id=args.right_wrist_cam_id)
    main_thread = threading.Thread(target=main_loop)
    # 启动主循环线程
    main_thread.start()

    # 等待服务器线程和主循环线程结束
    server_thread.join()

    wrist_camera_server_thread.join()
    main_thread.join()

    if should_stop:
        exit(0)
