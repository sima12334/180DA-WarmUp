import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
client.connect((host, 8080))
client.send("I am CLIENT\n")
from_server = client.recv(4096)
print(from_server)
client.close()
print(from_server)