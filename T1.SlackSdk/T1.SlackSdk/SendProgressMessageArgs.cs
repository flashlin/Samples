namespace T1.SlackSdk;

public class SendProgressMessageArgs
{
    public string ChannelId { get; set; } = string.Empty;
    public string Ts { get; set; } = string.Empty;
    public string ProgressMessage { get; set; } = string.Empty;
    public string Username { get; set; } = string.Empty;
}