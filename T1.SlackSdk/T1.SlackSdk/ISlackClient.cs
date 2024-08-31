using T1.Standard.Common;

namespace T1.SlackSdk;

public interface ISlackClient
{
    Task<List<SlackHistoryItem>> GetHistoryAsync(GetHistoryArgs getHistoryArgs );
    Task<SlackUser> GetUserInfoAsync(string userId);
}