import logging
import sys
from concurrent import futures
from pathlib import Path
import time
import threading

import grpc
from google.protobuf import empty_pb2

sys.path.append(str(Path(__file__).parent / "generated"))

import generated.UnBARableAI_pb2_grpc as unbarable_ai_pb2_grpc


class UnBARableAIService(unbarable_ai_pb2_grpc.UnBARableAIServiceServicer):
    def __init__(self, event_update_received: threading.Event):
        self._event_update_received = event_update_received
        self.last_request = None
        self._lock = threading.Lock()

    def handleEventUpdate(self, request, context):
        logging.info("Received handleEventUpdate call")

        with self._lock:
            self.last_request = request

        # Signal an BAR_Environment.reset(), dass ein Update angekommen ist
        self._event_update_received.set()

        return empty_pb2.Empty()


class UnBARableAIGRPCServer:
    def __init__(self):
        self.server = None
        self._event_update_received = threading.Event()
        self._service = UnBARableAIService(self._event_update_received)

    def start(self):
        """
        Startet den gRPC-Server auf localhost port 50051
        """
        logging.basicConfig(level=logging.DEBUG)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        unbarable_ai_pb2_grpc.add_UnBARableAIServiceServicer_to_server(
            self._service, self.server
        )
        self.server.add_insecure_port("localhost:50051")
        self.server.start()
        logging.info("Python gRPC server listening on localhost:50051")

    def stop(self, grace: float = 2.0):
        """
        Stoppt den gRPC-Server und lässt laufende RPCs optional auslaufen.
        """
        if self.server:
            logging.info("Stopping gRPC server...")
            self.server.stop(grace)

    # --- Neu: Steuerfunktionen für reset() ---
    def clear_event_update(self):
        self._event_update_received.clear()

    def wait_for_event_update(self, timeout: float | None = None) -> bool:
        """
        Wartet bis handleEventUpdate aufgerufen wurde.
        Returns True, wenn Event kam, sonst False bei Timeout.
        """
        return self._event_update_received.wait(timeout=timeout)

    def has_event_update(self) -> bool:
        return self._event_update_received.is_set()

    def get_last_request(self):
        return self._service.last_request
