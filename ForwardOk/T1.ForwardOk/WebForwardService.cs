using System.Collections.Concurrent;
using Grpc.Core;
using GrpcForwarderKit;
using Microsoft.Extensions.Logging;
using T1.ForwardOk.Sockets;
using T1.ForwardOk.Utils;

namespace T1.ForwardOk;

public class WebForwardService : GrpcForwarder.GrpcForwarderBase
{
    private readonly ILogger<WebForwardService> _logger;
    private readonly ConcurrentDictionary<string, ForwarderSession> _session = new();

    public WebForwardService(ILogger<WebForwardService> logger)
    {
        _logger = logger;
    }

    public int MaxBufferSize { get; init; } = 4 * 1024;

    public override async Task<ConnectReply> Connect(ConnectRequest request, ServerCallContext context)
    {
        var remoteAddress = await $"{request.ServerEndpoint}".ParseAddressesAsync()
            .FirstAsync().ConfigureAwait(false);
        var webForwardConnection = new WebForwardConnection();
        var destinationConnection = new TcpAsyncClient()
        {
            MaxBufferSize = MaxBufferSize
        };
        var session = new ForwarderSession()
        {
            Id = Guid.NewGuid().ToString(),
            Forwarder = new ForwarderSlim(webForwardConnection, destinationConnection),
        };
        _session[session.Id] = session;
        await destinationConnection.ConnectAsync(remoteAddress).ConfigureAwait(false);
        return new ConnectReply
        {
            ErrorCode = ForwardErrorCode.NotExists,
            ConnectId = session.Id,
        };
    }

    public override Task Subscribe(SubscribeRequest request, IServerStreamWriter<DataReply> responseStream,
        ServerCallContext context)
    {
        return Task.Run(() =>
        {
            if (!_session.TryGetValue(request.ConnectId, out var session))
            {
                responseStream.WriteAsync(new DataReply
                {
                    ErrorCode = ForwardErrorCode.NotExists,
                    ConnectId = request.ConnectId,
                });
                return;
            }

            var webForwardConnection = (WebForwardConnection) session.Forwarder.Source;
            while (true)
            {
                var buffer = new byte[1024];
                responseStream.WriteAsync(new DataReply
                {
                    ErrorCode = ForwardErrorCode.Success,
                    ConnectId = request.ConnectId,
                    Data = buffer.ToByteString(0, 100),
                });
            }
        });
    }
}