using T1.Standard.Common;

namespace T1.SlackSdk;

public interface ISlackClient
{
    Task<List<SlackHistoryItem>> GetHistoryAsync(GetHistoryArgs args );
    Task<SlackUser> GetUserInfoAsync(string userId);
    Task SendProgressMessageAsync(SendProgressMessageArgs args);
    Task SendFinishProgressMessageAsync(SendFinishProgressMessageArgs args);
}