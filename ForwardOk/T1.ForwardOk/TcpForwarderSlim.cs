using System.Net;
using System.Net.Sockets;
using Google.Protobuf.WellKnownTypes;
using T1.ForwardOk.Utils;

namespace T1.ForwardOk
{
    public class TcpForwarderSlim
    {
        private readonly Socket _mainSocket =
            new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);

        private CancellationTokenSource? _cancellationTokenSource;
        private readonly IRemoteForwardConnection _remoteWeb;

        public TcpForwarderSlim(IRemoteForwardConnection remoteWeb)
        {
            _remoteWeb = remoteWeb;
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
                    await _remoteWeb.StartRemoteForwardAsync(new RemoteForwardReq
                    {
                        ServerEndpoint = destination,
                    }).ConfigureAwait(false);
                    var source = new AsyncSocket(sourceSocket, BufferSize);
                    source.BeginReceive(OnSourceReceived, OnSourceClosed);
                }
            }).ConfigureAwait(false);
        }

        private Task OnSourceClosed()
        {
            _remoteWeb?.CloseAsync();
            return Task.CompletedTask;
        }

        private Task OnSourceReceived(byte[] buffer, int offset, int bytes)
        {
            _remoteWeb?.SendAsync(buffer, 0, bytes);
            return Task.CompletedTask;
        }
    }
}