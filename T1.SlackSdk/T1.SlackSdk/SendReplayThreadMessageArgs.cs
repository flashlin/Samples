namespace T1.SlackSdk;

/// <summary>
/// Arguments for sending a reply message to a specific thread
/// </summary>
public class SendReplayThreadMessageArgs
{
    /// <summary>
    /// Channel ID where the original message was sent
    /// </summary>
    public string ChannelId { get; set; } = string.Empty;

    /// <summary>
    /// Thread timestamp of the original message
    /// </summary>
    public string Ts { get; set; } = string.Empty;

    /// <summary>
    /// Reply message content
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Optional username for the reply
    /// </summary>
    public string Username { get; set; } = string.Empty;
} 