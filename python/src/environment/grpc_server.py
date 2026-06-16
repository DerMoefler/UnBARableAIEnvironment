import logging
import sys
from concurrent import futures
from pathlib import Path

import grpc
from google.protobuf import empty_pb2

sys.path.append(str(Path(__file__).parent / "generated"))

import generated.UnBARableAI_pb2_grpc as unbarable_ai_pb2_grpc


class UnBARableAIService(unbarable_ai_pb2_grpc.UnBARableAIServicer):
    def handleEventUpdate(self, request, context):
        logging.info("Received handleEventUpdate call")
        return empty_pb2.Empty()


class UnBARableAIGRPCServer:
    def serve() -> None:
        logging.basicConfig(level=logging.INFO)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        unbarable_ai_pb2_grpc.add_UnBARableAIServicer_to_server(UnBARableAIService(), server)
        server.add_insecure_port("[::]:50051")
        server.start()
        logging.info("Python gRPC server listening on 0.0.0.0:50051")
        server.wait_for_termination()

