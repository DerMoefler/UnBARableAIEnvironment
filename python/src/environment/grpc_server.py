import logging
import sys
from concurrent import futures
from pathlib import Path
import time

import grpc
from google.protobuf import empty_pb2

sys.path.append(str(Path(__file__).parent / "generated"))

import generated.UnBARableAI_pb2_grpc as unbarable_ai_pb2_grpc


class UnBARableAIService(unbarable_ai_pb2_grpc.UnBARableAIServicer):
    def handleEventUpdate(self, request, context):
        logging.info("Received handleEventUpdate call")
        return empty_pb2.Empty()


class UnBARableAIGRPCServer:
    def __init__(self):        
        self.server = None

    def start(self) :
        """
        Startet den gRPC-Server auf localhost port 50051

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logging.basicConfig(level=logging.DEBUG)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        unbarable_ai_pb2_grpc.add_UnBARableAIServicer_to_server(
            UnBARableAIService(), self.server
        )
        self.server.add_insecure_port("localhost:50051")
        self.server.start()
        logging.info("Python gRPC server listening on localhost:50051")

    def stop(self, grace: float = 2.0):
        """
        Stoppt den gRPC-Server und lässt laufende RPCs optional auslaufen.

        Parameters
        ----------
        grace : float
            Zeit in Sekunden, die laufende RPCs noch beenden dürfen.

        Returns
        -------
        None
        """
        if self.server:
            logging.info("Stopping gRPC server...")
            self.server.stop(grace)

if __name__ == "__main__":
    server = UnBARableAIGRPCServer()
    server.start()
    time.sleep(1000)  # Keep the server running for a while to allow testing