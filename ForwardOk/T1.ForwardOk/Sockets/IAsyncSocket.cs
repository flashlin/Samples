using System.Net;

namespace T1.ForwardOk.Sockets;

public interface IAsyncSocket
{
    Task ConnectAsync(IPEndPoint endpoint);
    void SetupReceive(ReceiveAsyncCallback receiveAsyncCallback, ClosedAsyncCallback closedAsyncCallback);
    Task CloseAsync();
    Task SendAsync(byte[] buffer, int offset, int bytes);
    Task SendMessageAsync(string message);
}