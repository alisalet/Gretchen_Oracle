import sys
import os
import time

base_dir = os.path.dirname(os.path.abspath(__file__))

gretchen_build_path = os.path.join(base_dir, 'lib', 'build', 'lib')

if gretchen_build_path not in sys.path:
    sys.path.insert(0, gretchen_build_path)

try:
    from gretchen.robot import Robot
    HAS_ROBOT_LIB = True
    print("yippie! Gretchen is working!")
except ImportError as e:
    print(f"Import error: {e}")
    HAS_ROBOT_LIB = False

class GretchenHardware:
    def __init__(self, motor_path='COM8', cam_path=0):
        try:
            self.robot = Robot(motor_path, cam_path)
            self.robot.start()
            # Вместо center() сразу поднимаем голову к человеку
            self.look_at_face()
        except Exception as e:
            print(f"Robot hardware connection failed: {e}. Running in simulation.")
            self.robot = None

    def look_at_face(self):
        """Абсолютное позиционирование: подъем головы к человеку."""
        if self.robot:
            self.robot.move(0.0, 0.5)
            time.sleep(0.5)

    def look_at_table(self):
        """Абсолютное позиционирование: наклон вниз к картам."""
        if self.robot:
            self.robot.move(0.0, -0.4)
            time.sleep(0.5)

    def nod(self):
        if self.robot:
            time.sleep(0.2)
            self.robot.down(0.4)
            time.sleep(0.2)
            self.robot.up(0.4)
            time.sleep(0.2)
            self.robot.down(0.4)
            time.sleep(0.2)
            self.robot.up(0.4)
            time.sleep(0.2)
            self.robot.down(0.4)
            time.sleep(0.2)
            self.robot.up(0.4)
            time.sleep(0.2)
            self.look_at_face()

    def shake(self):
        if self.robot:
            self.robot.left(0.3)
            time.sleep(0.2)
            self.robot.right(0.3)
            time.sleep(0.2)
            self.robot.left(0.3)
            time.sleep(0.2)
            self.robot.left(0.3)
            time.sleep(0.2)
            self.robot.right(0.3)
            time.sleep(0.2)
            self.robot.left(0.3)
            time.sleep(0.2)
            self.look_at_face()