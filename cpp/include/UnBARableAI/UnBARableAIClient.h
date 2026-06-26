#pragma once

#include <grpcpp/grpcpp.h>
#include <google/protobuf/empty.pb.h>

#include <memory>
#include <string>

#include "UnBARableAI.grpc.pb.h"

class UnBARableAIClient {
public:
    /**
     * @brief Erstellt einen gRPC-Client für den Python-Server.
     *
     * @param target Zieladresse des gRPC-Servers, z. B. "127.0.0.1:50051".
     */
    explicit UnBARableAIClient(const std::string& target = "127.0.0.1:50051");

    /**
     * @brief Sendet den RPC `handleEventUpdate` an den Python-Server.
     *
     * @param timeout_ms Timeout in Millisekunden für den RPC. (maximale Zeit, die dein Client bereit ist zu warten, bis eine Antwort vom Server kommt)
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
