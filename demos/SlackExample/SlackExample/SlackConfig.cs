public class SlackConfig
{
    public string Token { get; set; }
}

public static class DateTimeExtension
{
    public static long ToUnixTimeSeconds(this DateTime time)
    {
        var utcTime = time.ToUniversalTime();
        return new DateTimeOffset(utcTime).ToUnixTimeSeconds();
    }
}