using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Grpc.Net.Client;
using DemoServer.Services;

namespace DemoServerTest
{
    public class DemoServerApp : WebApplicationFactory<Program>
    {
        private GrpcChannel? _channel;
        private IGreeterGrpcClient? _grpcClient;

        protected override void ConfigureWebHost(IWebHostBuilder builder)
        {
            builder.ConfigureServices(services =>
            {
                // Additional test services can be configured here if needed
            });
        }

        public void Initialize()
        {
            // Create the test server and get the base address
            var client = CreateClient();
            var baseAddress = client.BaseAddress!;

            // Create gRPC channel for the test server
            _channel = GrpcChannel.ForAddress(baseAddress, new GrpcChannelOptions
            {
                HttpClient = client
            });

            // Create the gRPC client
            var greeterClient = new Greeter.GreeterClient(_channel);
            _grpcClient = new GreeterGrpcClient(greeterClient);
        }

        public IGreeterGrpcClient Client
        {
            get
            {
                if (_grpcClient == null)
                    throw new InvalidOperationException("Call Initialize() first");
                return _grpcClient;
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _channel?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
