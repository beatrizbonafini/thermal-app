import paramiko
import subprocess
from scp import SCPClient
import time

def run_command(command, client):
    stdin, stdout, stderr = client.exec_command(command)
    print(stdout.read().decode())
    print(stderr.read().decode())
    #return stdout.read().decode(), stderr.read().decode()

proxy_host ='macalan.c3sl.ufpr.br'
proxy_user = 'blbonafini'
proxy_password = 'Beatriz1/*'

target_host = 'thanos'
target_user = 'blbonafini'
target_password = proxy_password

filename_input = "imagem.jpg"
filename_output = "resultado.jpg"

INFERENCE_PATH = 'RWTH_MICE_PROJECT/detectron2/'
COMMAND_INFERENCE = f'/home/blbonafini/miniconda3/bin/python3 {INFERENCE_PATH}inferencia.py'
COMMAND_SCP_TO_MACALAN = f'scp .\{filename_input} {proxy_user}@{proxy_host}:' # executar na minha máquina
COMMAND_SCP_TO_THANOS = f'scp {filename_input} {target_user}@{target_host}:{INFERENCE_PATH}' # executar na macalan

proxy = paramiko.SSHClient()
proxy.set_missing_host_key_policy(paramiko.AutoAddPolicy())
proxy.connect(proxy_host, username=proxy_user, password=proxy_password)

transport = proxy.get_transport()
dest_addr = (target_host, 22)
local_addr = (proxy_host, 22)
channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)

target = paramiko.SSHClient()
target.set_missing_host_key_policy(paramiko.AutoAddPolicy())
target.connect(target_host, username=target_user, password=target_password, sock=channel)

print(COMMAND_SCP_TO_MACALAN)
print(COMMAND_SCP_TO_THANOS)
print(COMMAND_INFERENCE)

subprocess.run(COMMAND_SCP_TO_MACALAN, shell=True)
run_command(COMMAND_SCP_TO_THANOS, target)
run_command(COMMAND_INFERENCE, target)

#print(COMMAND_INFERENCE)
#stdin, stdout, stderr = target.exec_command(COMMAND_INFERENCE)
#stdin, stdout, stderr = target.exec_command("pwd")
#print('STDOUT:', stdout.read().decode())
#print('STDERR:', stderr.read().decode())

target.close()
proxy.close()