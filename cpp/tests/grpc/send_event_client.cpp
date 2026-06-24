#include "UnBARableAI/UnBARableAIClient.h"

#include <iostream>

int main() {
    UnBARableAIClient client("127.0.0.1:50051");

    if (client.HandleEventUpdate(1000)) {
        std::cout << "RPC successful\n";
        return 0;
    }

    std::cerr << "RPC failed: " << client.GetLastError() << "\n";
    return 1;
}
