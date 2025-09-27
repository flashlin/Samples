using System.Runtime.CompilerServices;
using System.Threading.Channels;

namespace CodeBoyLib.MQ;

public interface IProgressQueue
{
    Channel<string> GetOrCreateChannel(string jobId);
    IAsyncEnumerable<string> Consume(string jobId, CancellationToken ct);
}
