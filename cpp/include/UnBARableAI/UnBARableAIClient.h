#pragma once

#include <grpcpp/grpcpp.h>
#include <google/protobuf/empty.pb.h>

#include <memory>
#include <string>

#include "UnBARableAI/grpc_connection.hpp"

#include "UnBARableAI.grpc.pb.h"

class UnBARableAIClient {
public:
    using GrpcConnection = UnBARableAINS::GrpcConnection;

    /**
     * @brief Erstellt einen gRPC-Client für den Python-Server.
     *
     * @param connection GrpcConnection containing the port.
     * @param ip IP Address to connect to (defaults to localhost).
     */
    explicit UnBARableAIClient(const GrpcConnection& connection,
                               const std::string& ip = "127.0.0.1");

    /**
     * @brief Sendet den RPC `handleEventUpdate` an den Python-Server.
     *
     * @param timeout_ms Timeout in Millisekunden für den RPC. (maximale Zeit, die dein Client
     * bereit ist zu warten, bis eine Antwort vom Server kommt)
     * @return true, wenn der RPC erfolgreich war, sonst false.
     */
    bool HandleEventUpdate(int timeout_ms = 1000);

    /**
     * @brief Liefert die letzte Fehlermeldung des Clients.
     *
     * @return Letzte Fehlermeldung oder leerer String.
     */
    const std::string& GetLastError() const { return last_error_; }

    /**
     * @brief Liefert das aktuell konfigurierte Ziel.
     *
     * @return Zieladresse des Servers.
     */
    const std::string& GetTarget() const { return target_; }

private:
    std::string target_;
    std::string last_error_;

    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<UnBARableAIService::Stub> stub_;
};
