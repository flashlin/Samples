using DemoServer.Services;

namespace DemoServer.Services
{
    public class GreeterService : IGreeterGrpcService
    {
        public Task<HelloReplyGrpcDto> SayHello(HelloRequestGrpcDto request)
        {
            var response = new HelloReplyGrpcDto
            {
                Message = $"Hello {request.Name}!"
            };
            return Task.FromResult(response);
        }
    }
}
