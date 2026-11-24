#include "UnBARableAI/UnBARableAIClient.h"
#include "UnBARableAI/grpc_connection.hpp"

#include <iostream>

int main() {
    UnBARableAINS::GrpcConnection conn = {.port=50051};
    UnBARableAIClient client(conn);

    if (client.HandleEventUpdate(1000)) {
        std::cout << "RPC successful\n";
        return 0;
    }

    std::cerr << "RPC failed: " << client.GetLastError() << "\n";
    return 1;
}
