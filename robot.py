"""FLL drive base for LEGO SPIKE Prime (Pybricks v3+).

Port map
--------
Port.A  Left drive motor
Port.B  Right drive motor
Port.C  Front-left color sensor
Port.D  Front-right color sensor
IMU     Built-in hub gyro (heading is clockwise-positive)
"""

from pybricks.hubs import PrimeHub
from pybricks.parameters import Direction, Port
from pybricks.pupdevices import ColorSensor, Motor
from pybricks.tools import StopWatch, wait

# ---------------------------------------------------------------------------
# Mechanical constants — measure these on YOUR robot, then retune PD.
# SPIKE small tires are 56 mm; the larger FLL wheels are often 88 mm.
# ---------------------------------------------------------------------------
WHEEL_DIAMETER_MM = 56.0
# If forward makes the left wheel spin the wrong way, flip this Direction.
LEFT_MOTOR_DIRECTION = Direction.COUNTERCLOCKWISE
RIGHT_MOTOR_DIRECTION = Direction.CLOCKWISE

# ---------------------------------------------------------------------------
# Drive-straight PD (heading error -> deg/s of motor correction)
#   DRIVE_KP:  Raise if the robot still curves and does not come back to
#              heading. Lower if it weaves / oscillates while going straight.
#   DRIVE_KD:  Raise to damp weaving after a Kp increase. Lower if it feels
#              sluggish or if gyro noise makes the motors chatter.
# Start around Kp=4, Kd=0.4 on a 56 mm-wheel SPIKE Prime, then tune on mat.
# ---------------------------------------------------------------------------
DRIVE_KP = 4.0
DRIVE_KD = 0.4
DRIVE_SPEED_DEG_S = 400
DRIVE_MIN_SPEED_DEG_S = 80

# ---------------------------------------------------------------------------
# In-place turn PD (heading error -> deg/s of opposite wheel speeds)
#   TURN_KP:   Raise for snappier turns. Lower if it overshoots the target.
#   TURN_KD:   Raise to kill overshoot / oscillation at the end of the turn.
#              Lower if it never quite reaches the target (too much damping).
# Start around Kp=6, Kd=0.6, then tune 90° and 180° on the field.
# ---------------------------------------------------------------------------
TURN_KP = 6.0
TURN_KD = 0.6
TURN_MAX_SPEED_DEG_S = 450
TURN_MIN_SPEED_DEG_S = 60
TURN_ANGLE_TOLERANCE_DEG = 1.5
TURN_SETTLE_MS = 80

# Color-sensor square-up. FLL black lines usually reflect ~0–15%.
# Raise BLACK_REFLECTION_MAX if it never sees the line; lower if it
# false-triggers on dark gray mat.
BLACK_REFLECTION_MAX = 18
ALIGN_SPEED_DEG_S = 120
ALIGN_BACKUP_MM = 25
ALIGN_TIMEOUT_MS = 8000

IMU_READY_TIMEOUT_MS = 3000
CONTROL_LOOP_MS = 10
MAX_MOTOR_DEG_S = 1000


class Robot:
    """Differential-drive SPIKE Prime with IMU heading hold and line square-up."""

    def __init__(self):
        self.hub = PrimeHub()

        self.left_motor = Motor(Port.A, LEFT_MOTOR_DIRECTION)
        self.right_motor = Motor(Port.B, RIGHT_MOTOR_DIRECTION)

        self.left_color = ColorSensor(Port.C)
        self.right_color = ColorSensor(Port.D)

        self.left_motor.reset_angle(0)
        self.right_motor.reset_angle(0)
        self._wait_for_imu()
        self.hub.imu.reset_heading(0)

    def drive_straight_mm(self, distance_mm, speed_deg_s=DRIVE_SPEED_DEG_S):
        """Drive ``distance_mm`` (negative = reverse) with IMU heading PD.

        Holds the heading captured at the start of the move so the robot
        does not accumulate yaw from wheel slip or uneven motors.
        """
        if distance_mm == 0:
            return

        target_heading = self.hub.imu.heading()
        target_deg = abs(self._mm_to_motor_deg(distance_mm))
        direction = 1 if distance_mm > 0 else -1
        speed_deg_s = abs(speed_deg_s)

        self.left_motor.reset_angle(0)
        self.right_motor.reset_angle(0)

        last_error = 0.0
        timer = StopWatch()
        first_loop = True

        while self._average_motor_deg() < target_deg:
            remaining = target_deg - self._average_motor_deg()
            # Slow down in the last ~90° of wheel rotation so we do not
            # overshoot the millimeter target.
            ramp = min(1.0, remaining / 90.0)
            base = direction * max(DRIVE_MIN_SPEED_DEG_S, speed_deg_s * ramp)

            error = self._heading_error(target_heading)
            dt = max(timer.time() / 1000, 0.001)
            derivative = 0.0 if first_loop else (error - last_error) / dt
            # PD: correction is extra deg/s applied as a yaw (left vs right).
            correction = DRIVE_KP * error + DRIVE_KD * derivative

            self.left_motor.run(self._clamp_speed(base - correction))
            self.right_motor.run(self._clamp_speed(base + correction))

            last_error = error
            first_loop = False
            timer.reset()
            wait(CONTROL_LOOP_MS)

        self._hold()

    def turn_angle(self, target_heading_deg):
        """Turn in place until IMU heading matches ``target_heading_deg``.

        Heading is absolute (0 is the heading from ``__init__`` / last
        ``reset_heading``). Positive is clockwise, matching Pybricks IMU.
        Example: ``robot.turn_angle(90)`` faces +90° from start.
        """
        last_error = 0.0
        timer = StopWatch()
        first_loop = True
        settled = StopWatch()
        at_target = False

        while True:
            error = self._heading_error(target_heading_deg)

            if abs(error) <= TURN_ANGLE_TOLERANCE_DEG:
                self._hold()
                if not at_target:
                    at_target = True
                    settled.reset()
                elif settled.time() >= TURN_SETTLE_MS:
                    break
                last_error = error
                first_loop = False
                timer.reset()
                wait(CONTROL_LOOP_MS)
                continue
            at_target = False

            dt = max(timer.time() / 1000, 0.001)
            derivative = 0.0 if first_loop else (error - last_error) / dt
            # Same PD form as drive: Kp*error + Kd*d(error)/dt.
            # Positive error (too far CCW of target) => clockwise wheel pair.
            command = TURN_KP * error + TURN_KD * derivative
            command = self._with_min_turn_speed(command)

            self.left_motor.run(self._clamp_speed(command))
            self.right_motor.run(self._clamp_speed(-command))

            last_error = error
            first_loop = False
            timer.reset()
            wait(CONTROL_LOOP_MS)

        self._hold()

    def auto_align(self, backup_mm=ALIGN_BACKUP_MM, speed_deg_s=ALIGN_SPEED_DEG_S):
        """Creep forward until both front sensors are on black, then square.

        Each wheel stops as soon as its sensor sees the line so a skewed
        approach still ends square. After both are on black, the robot
        holds and reverses ``backup_mm`` with heading PD.
        """
        speed_deg_s = abs(speed_deg_s)
        left_on_line = False
        right_on_line = False
        timeout = StopWatch()

        self.left_motor.run(speed_deg_s)
        self.right_motor.run(speed_deg_s)

        while not (left_on_line and right_on_line):
            if timeout.time() > ALIGN_TIMEOUT_MS:
                break

            if not left_on_line and self._sees_black(self.left_color):
                self.left_motor.hold()
                left_on_line = True
            if not right_on_line and self._sees_black(self.right_color):
                self.right_motor.hold()
                right_on_line = True

            wait(CONTROL_LOOP_MS)

        self._hold()
        if backup_mm:
            self.drive_straight_mm(-abs(backup_mm), speed_deg_s=speed_deg_s)

    def reset_heading(self, angle=0):
        """Set the IMU heading, e.g. after lining up on a wall or line."""
        self._hold()
        self.hub.imu.reset_heading(angle)

    def _wait_for_imu(self):
        timer = StopWatch()
        while not self.hub.imu.ready():
            if timer.time() > IMU_READY_TIMEOUT_MS:
                break
            wait(50)

    def _sees_black(self, sensor):
        return sensor.reflection() <= BLACK_REFLECTION_MAX

    def _heading_error(self, target_deg):
        """Smallest signed error, target minus current, in (-180, 180]."""
        error = target_deg - self.hub.imu.heading()
        return (error + 180) % 360 - 180

    def _mm_to_motor_deg(self, distance_mm):
        return distance_mm * 360.0 / (3.141592653589793 * WHEEL_DIAMETER_MM)

    def _average_motor_deg(self):
        return (abs(self.left_motor.angle()) + abs(self.right_motor.angle())) / 2

    def _with_min_turn_speed(self, command):
        if command == 0:
            return 0
        magnitude = min(TURN_MAX_SPEED_DEG_S, abs(command))
        if magnitude < TURN_MIN_SPEED_DEG_S:
            magnitude = TURN_MIN_SPEED_DEG_S
        return magnitude if command > 0 else -magnitude

    def _clamp_speed(self, speed):
        if speed > MAX_MOTOR_DEG_S:
            return MAX_MOTOR_DEG_S
        if speed < -MAX_MOTOR_DEG_S:
            return -MAX_MOTOR_DEG_S
        return speed

    def _hold(self):
        self.left_motor.hold()
        self.right_motor.hold()


# Mission stub — replace with your run. Keep the hub still for ~1 s so the
# IMU can finish calibrating before the first move.
robot = Robot()
wait(200)
# robot.drive_straight_mm(400)
# robot.turn_angle(90)
# robot.auto_align()
