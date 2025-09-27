using MassTransit;
using Microsoft.Extensions.Logging;

namespace CodeBoyLib.MQ;

public class StartJobConsumer : IConsumer<IStartJob>
{
    private readonly ILogger<StartJobConsumer> _logger;
    
    public StartJobConsumer(ILogger<StartJobConsumer> logger)
    {
        _logger = logger;
    }
    
    public async Task Consume(ConsumeContext<IStartJob> context)
    {
        var retryCount = context.GetRetryAttempt();
        if (retryCount > 0)
        {
            _logger.LogWarning("重新嘗試處理任務 ID: {JobId}，這是第 {RetryCount} 次重試。", 
                context.Message.JobId, retryCount);
        }
        await context.Message.Execute();
    }
}
