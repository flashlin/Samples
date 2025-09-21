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

        public Task<GetTimeReplyGrpcDto> SayTime(GetTimeRequestGrpcDto request)
        {
            return Task.FromResult(new GetTimeReplyGrpcDto
            {
                ReplyTime = request.StartTime.AddSeconds(1),
            });
        }

        public Task<GetIntListReplyGrpcDto> SayIntList()
        {
            return Task.FromResult(new GetIntListReplyGrpcDto
            {
                Ids = [1, 2, 3]
            });
        }

        public Task SayIntList2(GetIntListRequestGrpcDto request)
        {
            if (request.Ids[0] != 1)
            {
                throw new Exception($"Something went wrong with {request.Ids[0]}");
            }
            return Task.CompletedTask;
        }
    }
}
