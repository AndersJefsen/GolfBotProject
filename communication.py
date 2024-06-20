import paramiko
from scp import SCPClient

def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

#  EV3's IP address, username and password
ssh = create_ssh_client('192.168.11.243', 22, 'robot', 'maker')

with SCPClient(ssh.get_transport()) as scp:
    #scp.put('degrees.py', '/home/robot/golfbot_project')
    scp.put('/Users/andersjefsen/robotcode/main.py', '/home/robot/golfbot_project')
    #scp.put('/Users/brickrun -r -- pybricks-micropython /home/robot/golfbot_project/degrees-gyro.py andersjefsen/robotcode/main.py', '/home/robot')