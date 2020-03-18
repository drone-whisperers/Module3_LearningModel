import socket
import sys


class TestingClient:

    def run(self):
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the port where the server is listening
        server_address = ('localhost', 42069)
        print(f"connecting to localhost port 16668")
        sock.connect(server_address)

        try:
            # Send data
            while (True):
                print("message to send: ")
                message = input()

                if (message == "quit"):
                    break

                print(f"sending {message}")
                sock.sendall(message.encode())

                # Look for the response
                amount_received = 0
                amount_expected = len(message)

                data = sock.recv(1024).decode()
                print(f"received {data}")

        finally:
            print("closing socket")
            sock.close()