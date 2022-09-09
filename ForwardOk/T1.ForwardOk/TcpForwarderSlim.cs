using System.Net.Sockets;
using Google.Protobuf;
using Google.Protobuf.WellKnownTypes;
using Grpc.Net.Client;
using GrpcForwarderKit;
using T1.ForwardOk;
using T1.ForwardOk.Sockets;

namespace T1.ForwardOk
{
    public class TcpForwarderSlim
    {
        private readonly Socket _mainSocket =
            new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);

        private CancellationTokenSource? _cancellationTokenSource;
        private IRemoteForwardOk? _remote;

        public TcpForwarderSlim(IRemoteForwardOk remote)
        {
            _remote = remote;
        }

        public int BufferSize { get; set; } = 4 * 1024;

        public async Task StartAsync(string local, string destination)
        {
            var localEndpoint = await local.ParseAddressesAsync().FirstAsync().ConfigureAwait(false);
            _cancellationTokenSource = new CancellationTokenSource();
            _mainSocket.Bind(localEndpoint);
            _mainSocket.Listen(10);
            await Task.Run(async () =>
            {
                while (!_cancellationTokenSource.IsCancellationRequested)
                {
                    var sourceSocket = await _mainSocket.AcceptAsync().ConfigureAwait(false);
                    //_destination.Connect();
                    var source = new AsyncSocket(sourceSocket, BufferSize);
                    source.BeginReceive(OnSourceReceived, OnSourceClosed);
                }
            }).ConfigureAwait(false);
        }

        private void OnSourceClosed()
        {
            _remote?.Close();
        }

        private Task OnSourceReceived(byte[] buffer, int offset, int bytes)
        {
            _remote?.SendAsync(buffer, bytes);
        }
    }
}


public class GRpcClientAgent : IRemoteForwardOk
{
    private string _connectId = string.Empty;
    private GrpcForwarder.GrpcForwarderClient _client = null!;
    private CancellationTokenSource? _cancellationTokenSource;

    public void Connect(ConnectRequest req)
    {
        _cancellationTokenSource = new CancellationTokenSource();
        var channel = GrpcChannel.ForAddress($"http://{req.ServerName}:{req.ServerPort}");
        _client = new GrpcForwarder.GrpcForwarderClient(channel);
        var reply = _client.Connect(req);
        _connectId = reply.ConnectId;
    }

    public void BeginReceive(ReceiveAsyncCallback receiveAsyncCallback, Action closeCallback)
    {
        Task.Run(async () =>
        {
            using var call = _client.Subscribe(new SubscribeRequest()
            {
                ConnectId = _connectId
            });
            while (await call.ResponseStream.MoveNext(_cancellationTokenSource!.Token))
            {
                var reply = call.ResponseStream.Current;
                if (reply != null)
                {
                    receiveAsyncCallback(reply.Data.ToByteArray(), reply.Data.Length);
                }
            }
        });
    }

    public void Close()
    {
    }

    public void SendAsync(byte[] buffer, int bytes)
    {
        var data = new Span<byte>(buffer);
        _client.Send(new SendRequest()
        {
            ConnectId = _connectId,
            Data = ByteString.CopyFrom(data)
        });
    }

    public void SendMessage(string message)
    {
        var bytes = message.FastToBytes();
        SendAsync(bytes, bytes.Length);
    }
}