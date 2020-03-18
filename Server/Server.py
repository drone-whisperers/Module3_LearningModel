from Translator import Translator
from Classifier import Classifier
import socket

PORT = 42069
SERVER_NAME = ""


class Server:
    _classifier = None
    _translator = None
    _socket = None

    def __init__(self):
        self._classifier = Classifier()
        self._translator = Translator()

        # Create a TCP/IP socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind((SERVER_NAME, PORT))

        return

    def run(self):
        # Listen for incoming connections
        self._socket.listen(1)

        while True:
            print("Awaiting Request")
            connection, client_address = self._socket.accept()

            try:
                print(f"connection from {client_address}")

                # Receive the data in small chunks and retransmit it
                while True:
                    data = connection.recv(1024).decode()
                    print(f"received: {data}")
                    if data:
                        pred = self.interpret_command(data)
                        print(f"response: {pred}")
                        connection.sendall(pred.encode())
                    else:
                        print(f"no more data from {client_address}")
                        break

            finally:
                # Clean up the connection
                connection.close()
        return

    def interpret_command(self, command):
        classifications = self._classifier.classify_command(command)

        #if len(classifications) == 0:
        #    return_message = "command not recognized"
        #else:
        translation = self._translator.translate_command(command)
        return_message = translation

        return return_message

