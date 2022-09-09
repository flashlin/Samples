using System.Net;

namespace T1.ForwardOk.Sockets;

public class WebForwardConnection : IAsyncSocket
{
    private ReceiveAsyncCallback? _receiveAsyncCallback;
    private ClosedAsyncCallback? _closedAsyncCallback;

    public Task ConnectAsync(IPEndPoint endpoint)
    {
        throw new NotImplementedException();
    }

    public void SetupReceive(ReceiveAsyncCallback receiveAsyncCallback, ClosedAsyncCallback closedAsyncCallback)
    {
        _closedAsyncCallback = closedAsyncCallback;
        _receiveAsyncCallback = receiveAsyncCallback;
    }

    public Task CloseAsync()
    {
        throw new NotImplementedException();
    }

    public Task SendAsync(byte[] buffer, int offset, int bytes)
    {
        throw new NotImplementedException();
    }

    public Task SendMessageAsync(string message)
    {
        throw new NotImplementedException();
    }
}