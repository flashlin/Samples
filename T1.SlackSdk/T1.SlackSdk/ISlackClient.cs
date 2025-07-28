using T1.Standard.Common;

namespace T1.SlackSdk;

public interface ISlackClient
{
    Task<List<SlackHistoryItem>> GetHistoryAsync(GetHistoryArgs args );
    Task<SlackUser> GetUserInfoAsync(string userId);
    Task SendProgressMessageAsync(SendProgressMessageArgs args);
    Task SendFinishProgressMessageAsync(SendFinishProgressMessageArgs args);

    /// <summary>
    /// Replace user mentions in text with custom replacement function
    /// </summary>
    /// <param name="text">Original text containing user mentions</param>
    /// <param name="getLabelValueFn">Function to replace user mention with custom value</param>
    /// <returns>Text with replaced user mentions</returns>
    Task<string> ReplaceUserLabelTextAsync(string text, Func<string, Task<string>> getLabelValueFn);
}