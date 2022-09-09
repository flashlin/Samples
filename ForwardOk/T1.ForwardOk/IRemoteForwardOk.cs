using GrpcForwarderKit;
using T1.ForwardOk.Sockets;

namespace T1.ForwardOk;

public interface IRemoteForwardOk : IAsyncSocket
{
    void Connect(ConnectRequest req);
}