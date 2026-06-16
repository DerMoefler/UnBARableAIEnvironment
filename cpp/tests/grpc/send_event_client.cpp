#include <grpcpp/grpcpp.h>
#include <google/protobuf/empty.pb.h>
#include <iostream>
#include <memory>
#include <string>

#include "UnBARableAI.grpc.pb.h"

int main(int argc, char** argv) {
    std::string target = "127.0.0.1:50051";

    if (argc > 1) {
        target = argv[1];
    }

    auto channel = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    auto stub = UnBARableAI::NewStub(channel);

    grpc::ClientContext context;
    google::protobuf::Empty request;
    google::protobuf::Empty response;

    grpc::Status status = stub->handleEventUpdate(&context, request, &response);

    if (status.ok()) {
        std::cout << "gRPC call succeeded: handleEventUpdate sent to " << target << std::endl;
        return 0;
    }

    std::cerr << "gRPC call failed: " << status.error_message() << std::endl;
    return 1;
}
