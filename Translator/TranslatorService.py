from Translator import Translator
import socket

PORT = 42069
HOST = "localhost"
translator = Translator()

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.bind((HOST, PORT))
socket.listen()

while True:
    print("Awaiting Request")
    connection, client_address = socket.accept()

    try:
        print(f"connection from {client_address}")

        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(1024).decode()
            print(f"received: {data}")
            if data:
                translation = translator.translate_command(data)
                print(f"response: {translation}")
                connection.sendall(translation.encode())
            else:
                print(f"no more data from {client_address}")
                break

    finally:
        # Clean up the connection
        connection.close()
