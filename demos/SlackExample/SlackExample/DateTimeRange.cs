namespace SlackExample;

public class DateTimeRange
{
    public DateTime Start { get; init; }
    public DateTime End { get; init; }

    public static DateTimeRange Day(DateTime date)
    {
        var dayStart = date.Date;
        var dayEnd = dayStart.AddDays(1).Subtract(TimeSpan.FromMilliseconds(1));
        return new DateTimeRange
        {
            Start = dayStart,
            End = dayEnd
        };
    }
}