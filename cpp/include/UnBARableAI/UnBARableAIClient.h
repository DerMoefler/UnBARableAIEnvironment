#pragma once


#include <memory>
#include <string>


class UnBARableAIClient {
public:
    /**
     * @brief Erstellt einen gRPC-Client für den Python-Server.
     *
     * @param target Zieladresse des gRPC-Servers, z. B. "127.0.0.1:50051".
     */
    explicit UnBARableAIClient(const std::string& target = "127.0.0.1:50051");

    ~UnBARableAIClient();

    UnBARableAIClient(UnBARableAIClient&&) noexcept;    
    UnBARableAIClient& operator=(UnBARableAIClient&&) noexcept;    

    UnBARableAIClient(const UnBARableAIClient&) = delete;    
    UnBARableAIClient& operator=(const UnBARableAIClient&) = delete;
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
    const std::string& GetLastError() const;

    /**
     * @brief Liefert das aktuell konfigurierte Ziel.
     *
     * @return Zieladresse des Servers.
     */
    const std::string& GetTarget() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
