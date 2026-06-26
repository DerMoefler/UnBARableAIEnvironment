#include "UnBARableAI/UnBARableAIClient.h"

#include <chrono>
#include <utility>

UnBARableAIClient::UnBARableAIClient(const std::string& target)
    : target_(target),
      channel_(grpc::CreateChannel(target_, grpc::InsecureChannelCredentials())),
      stub_(UnBARableAIService::NewStub(channel_)) {
}

bool UnBARableAIClient::HandleEventUpdate(int timeout_ms) {
    last_error_.clear();

    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms)
    );

    google::protobuf::Empty request;
    google::protobuf::Empty response;

    grpc::Status status = stub_->handleEventUpdate(&context, request, &response);

    if (status.ok()) {
        return true;
    }

    last_error_ =
        "code=" + std::to_string(static_cast<int>(status.error_code())) +
        ", message=\"" + status.error_message() + "\"";

    return false;
}