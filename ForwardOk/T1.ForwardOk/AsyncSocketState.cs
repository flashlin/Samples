using System.Net.Sockets;
using T1.ForwardOk.Sockets;

namespace T1.ForwardOk;

public class AsyncSocketState
{
    public Socket Socket { get; init; } = null!;
    public byte[] Buffer { get; init; } = null!;
    public ReceiveAsyncCallback? ReceiveAsyncCallback { get; init; }
    public Action? ClosedCallback { get; init; }
}