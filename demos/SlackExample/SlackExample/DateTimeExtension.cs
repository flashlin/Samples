namespace SlackExample;

public static class DateTimeExtension
{
    // public static long ToUnixTimeSeconds(this DateTime time)
    // {
    //     var utcTime = time.ToUniversalTime();
    //     return new DateTimeOffset(utcTime).ToUnixTimeSeconds();
    // }
    
    public static string ToDisplayString(this DateTime time)
    {
        return time.ToString("yyyy-MM-dd HH:mm:ss");
    }
    
    // public static DateTime ToDateTime(this long timestamp)
    // {
    //     var epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
    //     return epoch.AddSeconds(timestamp);
    // }

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
        return dateTime;
    }
}