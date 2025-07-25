namespace T1.SlackSdk;

public class SendFinishProgressMessageArgs
{
    public string ChannelId { get; set; } = string.Empty;
    public string Ts { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public string FinishMessage { get; set; } =string.Empty;
}