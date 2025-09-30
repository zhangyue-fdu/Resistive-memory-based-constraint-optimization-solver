import cv2
import numpy as np
import socket
import argparse
import json
import time
import threading
from pynput import keyboard  # 使用pynput实现跨线程键盘监听


def recv_all(sock, length):
    """Receive exactly *length* bytes. Handle time-out and broken connection."""
    data = b''
    while len(data) < length:
        try:
            packet = sock.recv(length - len(data))
            if not packet:
                return None  # connection closed
            data += packet
        except socket.timeout:
            raise  # let caller handle the time-out
        except (ConnectionResetError, BrokenPipeError) as e:
            print(f"Receive error: {e}")
            return None
    return data


def send_all(sock, data):
    """确保完整发送数据（保持不变）"""
    sent = 0
    while sent < len(data):
        try:
            sent += sock.send(data[sent:])
        except (ConnectionResetError, BrokenPipeError) as e:
            print(f"Send failed: {e}")
            return False
    return True

def process_image(frame):
    # coords = [[np.random.rand() for _ in range(3)] for _ in range(5)]
    start_position = [0.4, 0, 0.2]
    pick_position = [0.5, 0, 0.05]
    target_position = [0.90, -0.33, 0.1]
    # Move to pick position ...
    t1 = [-0.111557205574127, -0.4911391603827171, 0.24728634569122496]
    t2 = [-0.16995995282562743, -0.49112519314183095, 0.24731156052948544]
    t3 = [-0.2155605421524656, -0.49112634588582127, 0.2473133128446342]
    t4 = [-0.21562874887923145, -0.49110261583511944, 0.1936491573862663]
    t5 = [-0.2156302077370741, -0.491117024725997, 0.13332339931111087]
    t6 = [-0.21561941179467933, -0.49108085388067296, 0.04485814552531253]
    t7 = [-0.21557167901514418, -0.4911604365268564, 0.13571318300993743]
    t8 = [-0.2155502107227259, -0.49114712586407777, 0.19609339642426085]
    t9 = [-0.21553696569058847, -0.4911583689183554, 0.2541638298225253]
    t10 = [-0.2855529511878334, -0.491147066242763, 0.2540868951643842]
    t11 = [-0.35676892945004546, -0.49113962467964106, 0.25407940815021296]
    t12 = [-0.35678540079963295, -0.4159202503690435, 0.2541060442003197]
    t13 = [-0.3567880659229844, -0.33233615778078146, 0.25413111255485676]
    t14 = [-0.3931695229424184, -0.3040297481089461, 0.2541337377401945]
    t15 = [-0.3932001566162351, -0.3039899147572065, 0.1913314720429276]
    t16 = [-0.39320884466873923, -0.30401495424258096, 0.09975179838826367]
    
    # Open gripper, release object ...
    # end_effector_pos = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20]
    end_effector_pos = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16]
    return end_effector_pos

class ImageProcessor:
    def __init__(self,test_mode):
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.listener = None
        self.test_mode = test_mode

        self.video_writer = None  # To hold the VideoWriter object
        self.is_recording = False


    def start_keyboard_listener(self, sock):
        """Spawn a background thread that reacts to key presses."""

        def on_press(key):
            try:
                if key.char == 'j':  # press “j” to capture
                    print("Key 'j' pressed.")
                    with self.frame_lock:  # Lock and get the current frame
                        current_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                    if current_frame is not None:
                        self.process_and_send(sock, current_frame)
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

    def process_and_send(self, sock, frame):
        """Run vision algorithm and send resulting coordinates."""
        
        # Generate coordinates (keep the original logic)
        coords = [[np.random.rand() for _ in range(3)] for _ in range(7)] if self.test_mode else process_image(frame)

        # Send data (with lock protection)
        with self.send_lock:
            data = json.dumps(coords).encode()
            header = len(data).to_bytes(4, 'big')
            if not send_all(sock, header + data):
                raise ConnectionError("发送失败")


def main(server_ip, server_port, test_mode):
    processor = ImageProcessor(test_mode)

    # Configure video encoder / output file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format encoder
    out = out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480)) # Initialize the video writing object


    while True:  # outer reconnect loop
        sock = None
        try:
            print('1')
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(610)
            print('2')
            sock.connect((server_ip, server_port))
            print(f"Connected to {server_ip}:{server_port}")

            # Start keyboard listener
            processor.start_keyboard_listener(sock)

            # Receive / display frames
            while True:
                try:
                    # Receive image data
                    header = recv_all(sock, 4)
                    if not header: break

                    length = int.from_bytes(header, 'big')
                    img_data = recv_all(sock, length)
                    if not img_data: break

                    # Decode and update the latest frame
                    frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                    frame_copy = frame.copy()
                    if frame is not None:
                        with processor.frame_lock:
                            processor.latest_frame = frame

                        cv2.imshow('Client View', frame_copy)
                        out.write(frame)
    
                        if cv2.waitKey(1) == 27:  # ESC to exit
                            raise KeyboardInterrupt

                except (socket.timeout, ConnectionError) as e:
                    print(f"Connection error: {e}")
                    break

        except (ConnectionRefusedError, KeyboardInterrupt) as e:
            print(f"Connection interrupted: {e}")
        finally:
            # 资源释放
            if processor.listener:
                processor.listener.stop()
            if sock:
                sock.close()
            cv2.destroyAllWindows()
            out.release()  # 释放视频写入对象
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='8.134.128.59')
    parser.add_argument('--port', type=int, default='6001')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    main(args.ip, args.port, args.test)