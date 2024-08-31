namespace T1.SlackSdk;

public class SlackMessage
{
    public SlackUser User { get; set; } = SlackUser.Empty;
    public string Text { get; set; } = string.Empty;
    public DateTime Time { get; set; }
}