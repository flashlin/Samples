using System.Net;
using Google.Protobuf;
using Google.Protobuf.WellKnownTypes;
using Grpc.Net.Client;
using Grpc.Net.Client.Web;
using GrpcForwarderKit;
using T1.ForwardOk.Sockets;
using T1.ForwardOk.Utils;

namespace T1.ForwardOk;

public class GrpcClientConnection : IRemoteForwardConnection
{
    private string _connectId = string.Empty;
    private GrpcForwarder.GrpcForwarderClient _client = null!;
    private CancellationTokenSource? _cancellationTokenSource;

    public async Task ConnectAsync(IPEndPoint endpoint)
    {
        var channel = GrpcChannel.ForAddress($"http://{endpoint.Address.ToString()}:{endpoint.Port}", new GrpcChannelOptions
        {
            HttpHandler = new GrpcWebHandler(new HttpClientHandler())
        });
        _client = new GrpcForwarder.GrpcForwarderClient(channel);
        await _client.ReadyAsync(new Empty()).ConfigureAwait(false);
    }

    public void SetupReceive(ReceiveAsyncCallback receiveAsyncCallback, ClosedAsyncCallback closeAsyncCallback)
    {
        Task.Run(async () =>
        {
            using var call = _client.Subscribe(new SubscribeRequest()
            {
                ConnectId = _connectId
            });
            try
            {
                while (await call.ResponseStream.MoveNext(_cancellationTokenSource!.Token).ConfigureAwait(false))
                {
                    var reply = call.ResponseStream.Current;
                    if (reply != null)
                    {
                        await receiveAsyncCallback(reply.Data.ToByteArray(), 0, reply.Data.Length)
                            .ConfigureAwait(false);
                    }
                }
            }
            catch
            {
                await closeAsyncCallback().ConfigureAwait(false);
            }
        });
    }

    public Task SendAsync(byte[] buffer, int offset, int bytes)
    {
        var data = new Span<byte>(buffer);
        _client.Send(new SendRequest()
        {
            ConnectId = _connectId,
            Data = ByteString.CopyFrom(data)
        });
        return Task.CompletedTask;
    }

    public Task SendMessageAsync(string message)
    {
        var bytes = message.FastToBytes();
        return SendAsync(bytes, 0, bytes.Length);
    }

    public Task StartRemoteForwardAsync(RemoteForwardReq req)
    {
        throw new NotImplementedException();
    }

    public async Task StartForwardAsync(RemoteForwardReq req)
    {
        _cancellationTokenSource = new CancellationTokenSource();
        var reply = await _client.ConnectAsync(new ConnectRequest
        {
            ServerEndpoint = req.ServerEndpoint,
        }).ConfigureAwait(false);
        _connectId = reply.ConnectId;
    }

    public Task CloseAsync()
    {
        return Task.CompletedTask;
    }
}