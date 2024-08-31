namespace T1.SlackSdk;

public class SlackHistoryItem
{
    public Guid Id { get; set; }
    public SlackUser User { get; set; } = SlackUser.Empty;
    public string Text { get; set; } = string.Empty;
    public DateTime Time { get; set; }
    public List<SlackMessage> ThreadMessages { get; set; } = [];
}