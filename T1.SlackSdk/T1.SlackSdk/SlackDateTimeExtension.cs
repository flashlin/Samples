namespace T1.SlackSdk;

public static class SlackDateTimeExtension
{
    public static string ToDisplayString(this DateTime time)
    {
        return time.ToString("yyyy-MM-dd HH:mm:ss");
    }
    
    public static string ToSlackTs(this DateTime dateTime)
    {
        var dateTimeOffset = new DateTimeOffset(dateTime.ToUniversalTime());
        var seconds = dateTimeOffset.ToUnixTimeSeconds();
        var milliseconds = dateTimeOffset.Millisecond;
        return $"{seconds}.{milliseconds:D3}";
    }
    
    public static DateTime SlackTsToDateTime(this string slackTs)
    {
        if (string.IsNullOrEmpty(slackTs))
        {
            return DateTime.UnixEpoch;
        }
        var parts = slackTs.Split('.');
        var seconds = long.Parse(parts[0]);
        var dateTime = DateTimeOffset.FromUnixTimeSeconds(seconds).UtcDateTime;
        if (parts.Length > 1)
        {
            var milliseconds = int.Parse(parts[1].PadRight(3, '0').Substring(0, 3));
            dateTime = dateTime.AddMilliseconds(milliseconds);
        }
        return dateTime.ToLocalTime();
    }
}