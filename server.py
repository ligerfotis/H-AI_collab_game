import json
import socket
import time


class Server:
    """
    TCP Socket Server implementation for a single client
    """

    def __init__(self):
        self.port = 6610
        self.buffer_size = 100024
        self.socket = None
        self.connection = None
        self.start_time = None

    def start(self):
        """
         Starts listening for connections

         #NOTE: Blocking until a client is connected
         """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.socket.bind(('', self.port))
        self.socket.listen()
        print('listening connections')
        self.connection, addr = self.socket.accept()
        self.start_time = time.time()

        print('socket accepted')

    def request(self, namespace: str, payload: dict = None) -> dict:
        """
        Sends a request to client and wait for response
        #NOTE: Blocking until the client response

        :param namespace: the duration of the agent's action on the game
        :param payload: the duration of the agent's action on the game
        :return: Client's response
        """

        if payload is None:
            self._send(str.encode(namespace + "\n"))  # sending -> "namespace"
        else:
            self._send(str.encode(namespace + "|" + json.dumps(payload) + "\n"))  # sending -> "namespace"|{payload}

        # if time.time() - self.start_time > 15:
        #     self.start_time = time.time()
        #     print("Close connection unexpected.")
        #     self.close()

        res = self._read_next().decode("utf-8")

        return json.loads(res)

    def close(self):
        """
        Closes the socket and connection
        """
        print('close')
        try:
            self.connection.close()
        except:
            pass
        try:
            self.socket.close()
        except:
            pass

    def _read_next(self):
        data = None

        while data is None:
            data = self.connection.recv(self.buffer_size)
        return data

    def _send(self, data):
        self.connection.send(data)


if __name__ == '__main__':
    server = Server()
    server.start()
    print(server.request('step', {'test': 'test'}))
    server.close()
