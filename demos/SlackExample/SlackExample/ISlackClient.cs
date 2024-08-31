namespace SlackExample;

public interface ISlackClient
{
    Task<List<SlackHistoryItem>> GetHistoryAsync(string channelId, DateTimeRange range);
}