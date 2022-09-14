using System.Net.Sockets;
using T1.ForwardOk.Sockets;

namespace T1.ForwardOk.Utils;

public class AsyncSocketState
{
    public Socket Socket { get; init; } = null!;
    public byte[] Buffer { get; init; } = null!;
    public ReceiveAsyncCallback ReceiveAsyncCallback { get; init; } = null!;
    public ClosedAsyncCallback ClosedCallback { get; init; } = null!;
}