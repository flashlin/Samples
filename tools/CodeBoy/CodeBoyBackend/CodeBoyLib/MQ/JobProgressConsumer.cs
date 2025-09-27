using MassTransit;

namespace CodeBoyLib.MQ;

public class JobProgressConsumer : IConsumer<JobProgressMessage>
{
    private readonly MassTransitProgressQueue _progressQueue;

    public JobProgressConsumer(MassTransitProgressQueue progressQueue)
    {
        _progressQueue = progressQueue;
    }

    public async Task Consume(ConsumeContext<JobProgressMessage> context)
    {
        var channel = _progressQueue.GetOrCreateChannel(context.Message.JobId);
        await channel.Writer.WriteAsync(context.Message.Content);
    }
}
