using T1.ForwardOk.Sockets;

namespace T1.ForwardOk;

public class ForwarderSlim
{
    private readonly IAsyncSocket _source;
    private readonly IAsyncSocket _destination;

    public ForwarderSlim(IAsyncSocket source, IAsyncSocket destination)
    {
        _destination = destination;
        _source = source;
        source.SetupReceive(OnSourceReceived, OnSourceClosed);
        destination.SetupReceive(OnDestinationReceived, OnDestination);
    }

    public IAsyncSocket Source => _source;
    public IAsyncSocket Destination => _destination;

    public Task SendToDestinationAsync(byte[] buffer, int offset, int length)
    {
        return _destination.SendAsync(buffer, offset, length);
    }

    private async Task OnDestination()
    {
        await _destination.CloseAsync().ConfigureAwait(false);
        await _source.CloseAsync().ConfigureAwait(false);
    }

    private Task OnDestinationReceived(byte[] buffer, int offset, int bytes)
    {
        return _source.SendAsync(buffer, offset, bytes);
    }

    private async Task OnSourceClosed()
    {
        await _source.CloseAsync().ConfigureAwait(false);
        await _destination.CloseAsync().ConfigureAwait(false);
    }

    private Task OnSourceReceived(byte[] buffer, int offset, int bytes)
    {
        return _destination.SendAsync(buffer, offset, bytes);
    }
}