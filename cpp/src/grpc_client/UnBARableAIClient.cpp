#include "UnBARableAI/UnBARableAIClient.h"

#include <grpcpp/grpcpp.h>
#include <google/protobuf/empty.pb.h>
#include "UnBARableAI.grpc.pb.h"

class UnBARableAIClient::Impl {
public:
    explicit Impl(const std::string& target)
        : target_(target),
          channel_(grpc::CreateChannel(target, grpc::InsecureChannelCredentials())),
          stub_(UnBARableAI::NewStub(channel_)) {}

    std::string target_;
    std::string last_error_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<UnBARableAI::Stub> stub_;
};

UnBARableAIClient::UnBARableAIClient(const std::string& target)
    : impl_(std::make_unique<Impl>(target)) {}

UnBARableAIClient::~UnBARableAIClient() = default;
UnBARableAIClient::UnBARableAIClient(UnBARableAIClient&&) noexcept = default;
UnBARableAIClient& UnBARableAIClient::operator=(UnBARableAIClient&&) noexcept = default;

bool UnBARableAIClient::HandleEventUpdate(int timeout_ms) {
    google::protobuf::Empty request;
    google::protobuf::Empty response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms));

    grpc::Status status = impl_->stub_->handleEventUpdate(&context, request, &response);
    if (!status.ok()) {
        impl_->last_error_ = status.error_message();
        return false;
    }
    impl_->last_error_.clear();
    return true;
}

const std::string& UnBARableAIClient::GetLastError() const { return impl_->last_error_; }
const std::string& UnBARableAIClient::GetTarget() const { return impl_->target_; }
