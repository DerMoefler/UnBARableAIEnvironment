import logging
import sys
from concurrent import futures
from pathlib import Path
import time
import threading
import time
from typing import Optional

import grpc
from google.protobuf import empty_pb2

sys.path.append(str(Path(__file__).parent / "generated"))
import generated.UnBARableAI_pb2_grpc as unbarable_ai_pb2_grpc


class UnBARableAIService(unbarable_ai_pb2_grpc.UnBARableAIServiceServicer):
    def __init__(self):
        self._condition = threading.Condition()

        # Anzahl empfangener handleEventUpdate()-RPCs
        self.arrived_update_count = 0

        # Höchste Update-ID, die von env.step() freigegeben wurde
        self.acked_update_count = 0
        self._stopping = False

    def handleEventUpdate(self, _request, _context):
        """
        Wird von dem UnBARableAIClient aufgerufen wenn die Observation ins shared memory geschrieben wurde.

        1. weckt reset()/step() auf
        2. blockiert selbst, bis reset()/step() ack_update() aufgerufen hat

        Parameters
        ----------
        request : google.protobuf.empty_pb2.Empty
            Leere, von gRPC deserialisierte Protocol-Buffer-Nachricht des UnBARableAIClient. 
            Wird von der API benötigt aber nicht verwendet.
        
        context : grpc.ServicerContext
            Serverseitiger Kontext des aktuellen gRPC-Aufrufs.
            Wird von der API benötigt aber nicht verwendet.
        
        Returns
        -------
        google.protobuf.empty_pb2.Empty
            Leere Protocol-Buffer-Antwort an den UnBARableAIClient.
            Wird von der API benötigt aber nicht verwendet.

        """
        logging.info("Received handleEventUpdate call")

        with self._condition:

            # Neues Update registrieren
            self.arrived_update_count += 1
            my_update_id = self.arrived_update_count

            logging.info(
                "handleEventUpdate arrived: update_id=%s (acked=%s)",
                my_update_id,
                self.acked_update_count,
            )

            # reset()/step() wecken
            self._condition.notify_all()

            # Jetzt selbst blockieren
            ok =self._condition.wait_for(
                lambda: self._stopping or self.acked_update_count >= my_update_id,
                timeout=10
            )

            if not ok:
                logging.warning(
                    "Timeout while waiting for ack: update_id=%s (acked=%s)",
                    my_update_id,
                    self.acked_update_count,
                )
            elif self._stopping:
                logging.info(
                    "Stopping while waiting for ack: update_id=%s (acked=%s)",
                    my_update_id,
                    self.acked_update_count,
                )         

            logging.info(
                "handleEventUpdate returning: update_id=%s (acked=%s)",
                my_update_id,
                self.acked_update_count,
            )

        return empty_pb2.Empty()

    # ------------------------------------------------------------
    # API für BAR_Environment
    # ------------------------------------------------------------


    def wait_for_next_update(
        self,
        previous_count: int,
        timeout: Optional[float] = None,
    ) -> tuple[str, Optional[int]]:
        """
        Wartet bis ein neues handleEventUpdate angekommen ist oder der Grpc Server gestoppt wurde.

        Parameters
        ----------
        previous_count : int
            Der zuletzt bekannte arrived_update_count.
            Gewartet wird auf arrived_update_count > previous_count.
        timeout : float | None
            Zeit in Sekunden, die maximal gewartet werden soll.
            None bedeutet unendlich lang warten.

        Returns
        -------
        status : str
            "update" wenn ein neues Update kam,
            "timeout" wenn die Wartezeit abgelaufen ist,
            "stopped" wenn der Grpc Server gestoppt wurde.
        arrived_update_count: int | None
            Anzahl empfangener handleEventUpdate()-RPCs, wenn ein neues Update kam.
            None bei Timeout.
        """
        with self._condition:
            ok = self._condition.wait_for(
                lambda: self._stopping or self.arrived_update_count > previous_count,
                timeout=timeout,
            )

            if not ok:
                return "timeout", None
            if self._stopping:
                return "stopped", None

            return "update", self.arrived_update_count

    def ack_update(self, update_id: int) -> None:
        """
        Gibt ein bestimmtes Update frei.

        Wird von step() aufgerufen, nachdem die action ins shared memory geschrieben wurde. Nach dem Aufruf von ack_update() darf der handleEventUpdate()-Call zurückkehren, sodass die Engine weiterlaufen kann.

        Parameters
        ----------
        update_id : int
            Die Update-ID, die freigegeben werden soll.
        
        Wichtig:
        - ack_update(1) gibt handleEventUpdate #1 frei, ack_update(2) gibt #2 frei, usw.
        """
        with self._condition:
            if update_id > self.acked_update_count:
                self.acked_update_count = update_id
                logging.info(
                    "Acked update_id=%s (arrived=%s)",
                    self.acked_update_count,
                    self.arrived_update_count,
                )
                self._condition.notify_all()

    def stop_waiters(self) -> None:
        """
        Weckt alle wartenden Threads auf, sodass sie erkennen, dass der Grpc Server gestoppt wurde.
        """
        with self._condition:
            self._stopping = True
            self._condition.notify_all()


class UnBARableAIGRPCServer:
    def __init__(self):
        self.server = None
        self._service = UnBARableAIService()

    def start(self):
        logging.basicConfig(level=logging.DEBUG)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        unbarable_ai_pb2_grpc.add_UnBARableAIServiceServicer_to_server(
            self._service,
            self.server,
        )
        self.server.add_insecure_port("localhost:50051")
        self.server.start()
        logging.info("Python gRPC server listening on localhost:50051")

    # ------------------------------------------------------------
    # Öffentliche API für BAR_Environment
    # ------------------------------------------------------------

    def wait_for_next_update(self, previous_count: int, timeout: Optional[float] = None,) -> tuple[str, Optional[int]]:
        return self._service.wait_for_next_update(
            previous_count=previous_count,
            timeout=timeout,
        )

    def ack_update(self, update_id: int) -> None:
        self._service.ack_update(update_id)

    def stop(self, grace: float = 2.0) -> None:
            self._service.stop_waiters()
    
            if self.server:
                logging.info("Stopping gRPC server...")
                self.server.stop(grace)
