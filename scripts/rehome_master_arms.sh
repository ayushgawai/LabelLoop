#!/bin/bash
# Fix "Homing failed: motion planner status error" / MoveIt START_STATE_IN_COLLISION (master_*_arm_link vs master_*_joy_trigger):
# the teleop controller arms were parked bent. Sends one slow (8 s) trajectory moving all 12 master-arm joints to 0 (rest) through
# four_arm_joint_trajectory_controller while holding the robot arms where they are. Hands off the controller arms while it runs.
SSH_ASKPASS=${ROBOT_ASKPASS:-$HOME/.config/labelloop/robot_askpass.sh} SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -o ConnectTimeout=8 xr@192.168.1.21 'docker exec desktop_client-robot_client-1 bash -c "source /opt/xr/bot/x2robot_client/share/x2robot_client/scripts/session_env.sh >/dev/null 2>&1; timeout 40 python3 - <<PY
import rclpy, time
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
rclpy.init(); n=rclpy.create_node(\"master_rehome\")
c=n.create_client(GetParameters, \"/four_arm_joint_trajectory_controller/get_parameters\"); c.wait_for_service(timeout_sec=5)
f=c.call_async(GetParameters.Request(names=[\"joints\"])); rclpy.spin_until_future_complete(n,f,timeout_sec=5)
joints=list(f.result().values[0].string_array_value)
got=[]; n.create_subscription(JointState, \"/joint_states\", lambda m: got.append(m), 10)
t=time.time()
while not got and time.time()-t<5: rclpy.spin_once(n, timeout_sec=0.2)
pos=dict(zip(got[-1].name, got[-1].position))
cur=[pos[j] for j in joints]; tgt=[0.0 if j.startswith(\"master\") else pos[j] for j in joints]
pub=n.create_publisher(JointTrajectory, \"/four_arm_joint_trajectory_controller/joint_trajectory\", 10); time.sleep(1.0)
pub.publish(JointTrajectory(joint_names=joints, points=[JointTrajectoryPoint(positions=cur, time_from_start=Duration(nanosec=200000000)), JointTrajectoryPoint(positions=tgt, time_from_start=Duration(sec=8))]))
t=time.time()
while time.time()-t<10: rclpy.spin_once(n, timeout_sec=0.2)
got.clear(); t=time.time()
while not got and time.time()-t<4: rclpy.spin_once(n, timeout_sec=0.2)
p=dict(zip(got[-1].name, got[-1].position)); print(\"master joints now\", [round(p[j],3) for j in joints if j.startswith(\"master\")])
PY"' 2>&1 | grep -v -i passw | tail -2
