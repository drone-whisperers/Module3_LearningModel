import sys

sys.path.insert(0, 'Translator')
sys.path.insert(0, 'Classifier')

from Server import Server

server = Server()
server.run()
