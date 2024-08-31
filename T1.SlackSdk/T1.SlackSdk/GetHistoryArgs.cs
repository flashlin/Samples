using T1.Standard.Common;

namespace T1.SlackSdk;

public class GetHistoryArgs
{
    public string ChannelId { get; set; } = string.Empty;
    public DateTimeRange Range { get; set; } = new();
    public int Limit { get; set; } = 100;
}