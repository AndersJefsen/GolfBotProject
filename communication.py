import paramiko
from scp import SCPClient

def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

# replace with your EV3's IP address, username and password
ssh = create_ssh_client('192.168.156.243', 22, 'robot', 'maker')

# replace 'local_file.txt' with your file path
# replace '/home/robot/' with the destination directory in the EV3
with SCPClient(ssh.get_transport()) as scp:
    #scp.put('degrees.py', '/home/robot/golfbot_project')
    #scp.put('calibrate_forwards_backwards.py', '/home/robot/golfbot_project')
    scp.put('degrees-gyro.py', '/home/robot/golfbot_project')