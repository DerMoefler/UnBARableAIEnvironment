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

        # Für sauberes Shutdown / Unblock
        self._stopping = False

    def handleEventUpdate(self, request, context):
        logging.info("Received handleEventUpdate call")

        with self._condition:
            self.last_request = request

            # Neues Update registrieren
            self.arrived_update_count += 1
            my_update_id = self.arrived_update_count

            logging.info(
                "handleEventUpdate arrived: update_id=%s (acked=%s)",
                my_update_id,
                self.acked_update_count,
            )

            # reset()/step() wecken, die gerade auf das NÄCHSTE Update warten
            self._condition.notify_all()

            # Jetzt selbst blockieren, bis dieses Update freigegeben wird
            while (
                not self._stopping
                and self.acked_update_count < my_update_id
            ):
                # Falls der Client-Call bereits cancelled / timed out ist:
                if not context.is_active():
                    logging.warning(
                        "RPC no longer active while waiting for ack of update_id=%s",
                        my_update_id,
                    )
                    break

                self._condition.wait(timeout=0.1)

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
    ) -> Optional[int]:
        """
        Wartet bis mindestens ein neues handleEventUpdate angekommen ist.

        Parameters
        ----------
        previous_count : int
            Der zuletzt bekannte arrived_update_count.
            Gewartet wird auf arrived_update_count > previous_count.
        timeout : float | None
            Timeout in Sekunden.

        Returns
        -------
        int | None
            Die neue Update-ID (arrived_update_count), wenn ein neues Update kam.
            None bei Timeout.
        """
        with self._condition:
            ok = self._condition.wait_for(
                lambda: self._stopping or self.arrived_update_count > previous_count,
                timeout=timeout,
            )

            if not ok or self._stopping:
                return None

            return self.arrived_update_count

    def ack_update(self, update_id: int) -> None:
        """
        Gibt ein bestimmtes Update frei, sodass der blockierte
        handleEventUpdate()-Call zurückkehren darf.

        Wichtig:
        - update_id muss die zuletzt von reset()/step() beobachtete Update-ID sein.
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

    def stop(self, grace: float = 2.0):
        self._service.stop_waiters()

        if self.server:
            logging.info("Stopping gRPC server...")
            self.server.stop(grace)

    # ------------------------------------------------------------
    # Öffentliche API für BAR_Environment
    # ------------------------------------------------------------

    def wait_for_next_update(
        self,
        previous_count: int,
        timeout: Optional[float] = None,
    ) -> Optional[int]:
        return self._service.wait_for_next_update(
            previous_count=previous_count,
            timeout=timeout,
        )

    def ack_update(self, update_id: int) -> None:
        self._service.ack_update(update_id)
