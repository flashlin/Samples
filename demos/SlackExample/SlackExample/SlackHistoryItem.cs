namespace SlackExample;

public class SlackHistoryItem
{
    public SlackUser User { get; set; } = SlackUser.Empty;
    public string Text { get; set; } = string.Empty;
    public DateTime Time { get; set; }
    public List<SlackMessage> ThreadMessages { get; set; } = [];
}