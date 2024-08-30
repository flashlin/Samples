namespace SlackExample;

public interface ISlackClient
{
    IAsyncEnumerable<SlackHistoryItem> GetHistoryAsync(string channelId, DateTimeRange range);
}