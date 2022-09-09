using T1.ForwardOk.Sockets;

namespace T1.ForwardOk;

public class ForwarderSession
{
    public string Id { get; init; } = null!;
    public ForwarderSlim Forwarder { get; init; } = null!;
}