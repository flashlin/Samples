using System.Collections.Concurrent;
using System.Threading.Channels;

namespace CodeBoyLib.MQ;

public class MassTransitProgressQueue : IProgressQueue
{
    private readonly ConcurrentDictionary<string, Channel<string>> _channels = new();

    public Channel<string> GetOrCreateChannel(string jobId)
        => _channels.GetOrAdd(jobId, _ => Channel.CreateUnbounded<string>());

    public async IAsyncEnumerable<string> Consume(string jobId, CancellationToken ct)
    {
        var channel = GetOrCreateChannel(jobId);

        await foreach (var msg in channel.Reader.ReadAllAsync(ct))
        {
            yield return msg;
        }
    }
}
