using System.Buffers;
using System.Net;
using System.Net.Sockets;

namespace T1.ForwardOk.Sockets;

public delegate Task ReceiveAsyncCallback(byte[] buffer, int offset, int length);

public delegate Task ClosedAsyncCallback();

public class TcpAsyncClient : IAsyncSocket
{
    private TcpClient? _tcp;
    private readonly byte[] _readBuffer;
    private readonly byte[] _writeBuffer;
    private ReceiveAsyncCallback? _receiveAsyncCallback;
    private CancellationTokenSource? _cancellationTokenSource;
    private NetworkStream? _stream;
    private ClosedAsyncCallback? _closedAsyncCallback;

    public TcpAsyncClient()
    {
        _readBuffer = ArrayPool<byte>.Shared.Rent(MaxBufferSize);
        _writeBuffer = ArrayPool<byte>.Shared.Rent(MaxBufferSize);
    }

    public int MaxBufferSize { get; init; } = 4 * 1024;

    public async Task ConnectAsync(IPEndPoint endpoint)
    {
        _cancellationTokenSource = new CancellationTokenSource();
        _tcp = new TcpClient() {NoDelay = true};
        _tcp.Client.DualMode = true;
        
        //_tcp.BeginConnect(endpoint.Address, endpoint.Port, EndConnectCallback, null);
        await _tcp.ConnectAsync(endpoint, _cancellationTokenSource.Token).ConfigureAwait(false);
        _stream = _tcp.GetStream();
        StartReadTask();
    }

    public void SetupReceive(ReceiveAsyncCallback receiveAsyncCallback, ClosedAsyncCallback closedAsyncCallback)
    {
        _receiveAsyncCallback = receiveAsyncCallback;
        _closedAsyncCallback = closedAsyncCallback;
    }

    public async Task CloseAsync()
    {
        _tcp!.Close();
        if (_closedAsyncCallback != null)
            await _closedAsyncCallback().ConfigureAwait(false);
    }

    public Task SendAsync(byte[] buffer, int offset, int length)
    {
        return _stream!.WriteAsync(buffer, offset, length, _cancellationTokenSource!.Token);
    }

    public Task SendMessageAsync(string message)
    {
        var bytes = message.FastToBytes();
        return SendAsync(bytes, 0, bytes.Length);
    }

    private void EndConnectCallback(IAsyncResult ar)
    {
        try
        {
            _tcp!.EndConnect(ar);
            _stream = _tcp.GetStream();
            StartReadTask();
        }
        catch (Exception ex)
        {
            _tcp?.Close();
            _tcp = null;
        }
    }

    private void StartReadTask()
    {
        var networkReadTask = Task.Run(async () =>
        {
            while (!_cancellationTokenSource!.IsCancellationRequested)
            {
                try
                {
                    var readLength = await _stream!
                        .ReadAsync(_readBuffer, 0, _readBuffer.Length, _cancellationTokenSource.Token)
                        .ConfigureAwait(false);
                    if (readLength <= 0) continue;
                    if (_receiveAsyncCallback != null)
                    {
                        await _receiveAsyncCallback(_readBuffer, 0, _readBuffer.Length).ConfigureAwait(false);
                    }
                }
                catch
                {
                    _tcp?.Close();
                }
            }
        }, _cancellationTokenSource!.Token).ConfigureAwait(false);
    }
}