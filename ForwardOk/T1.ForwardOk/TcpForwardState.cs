using System.Net.Sockets;
using T1.ForwardOk.Sockets;

namespace T1.ForwardOk;

public struct TcpForwardState
{
    public TcpForwardState(Socket sourceSocket, byte[] buffer, IAsyncSocket destination)
    {
        SourceSocket = sourceSocket;
        Buffer = buffer;
        Destination = destination;
    }

    public Socket SourceSocket { get; }
    public byte[] Buffer { get; }
    public IAsyncSocket Destination { get; }
}