using System;
using System.Threading.Tasks;
using Grpc.Core;
using ConsoleTestApp;

namespace ConsoleTestApp.GrpcServices
{
    /// <summary>
    /// gRPC 服務實現，基於 IGreeter
    /// </summary>
    public class GreeterGrpcService : Greeter.GreeterBase
    {
        private readonly IGreeter _greeter;

        public GreeterGrpcService(IGreeter greeter)
        {
            _greeter = greeter;
        }

        public override async Task<SayHelloAsyncReplyMessage> SayHelloAsync(SayHelloAsyncRequestMessage request, ServerCallContext context)
        {
            var name = request.Name;
            var result = await _greeter.SayHelloAsync(name);
            return new SayHelloAsyncReplyMessage { Result = result };
        }

        public override async Task<GetUserInfoAsyncReplyMessage> GetUserInfoAsync(GetUserInfoAsyncRequestMessage request, ServerCallContext context)
        {
            var id = request.Id;
            var includeDetails = request.IncludeDetails;
            var result = await _greeter.GetUserInfoAsync(id, includeDetails);
            return new GetUserInfoAsyncReplyMessage { Result = result };
        }

        public override async Task<AddReplyMessage> Add(AddRequestMessage request, ServerCallContext context)
        {
            var a = request.A;
            var b = request.B;
            var result = _greeter.Add(a, b);
            return new AddReplyMessage { Result = result };
        }

    }
}
