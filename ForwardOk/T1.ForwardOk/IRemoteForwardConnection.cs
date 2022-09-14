using GrpcForwarderKit;
using T1.ForwardOk.Sockets;

namespace T1.ForwardOk;

public interface IRemoteForwardConnection : IAsyncSocket
{
    Task StartRemoteForwardAsync(RemoteForwardReq req);
}