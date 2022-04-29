using SqliteCli.Repos;

namespace SqliteCli.Helpers;

public static class DateTimeExtension
{
    public static bool IsNowClosingTime(this DateTime time)
    {
        if (time.ToDate() != DateTime.Now.ToDate())
        {
            return true;
        }
        var openingTime = DateTime.Parse(time.ToDate().ToDateString() + " 09:00:00");
        var closingTime = DateTime.Parse(time.ToDate().ToDateString() + " 13:33:00");
        if (DateTime.Now < closingTime)
        {
            return false;
        }
        return true;
    }
    
    public static DateRange GetDateRange(this DateTime date)
    {
        return new DateRange
        {
            StartDate = date.StartOfMonth(),
            EndDate = date.EndOfMonth()
        };
    }

    public static DateTime StartOfMonth(this DateTime date)
    {
        return new DateTime(date.Year, date.Month, 1, 0, 0, 0);
    }

    public static DateTime EndOfMonth(this DateTime date)
    {
        return date.StartOfMonth().AddMonths(1).AddSeconds(-1);
    }

    public static bool IsWorkDay(this DateTime date)
    {
        if (date.DayOfWeek == DayOfWeek.Sunday)
            return false;
        if (date.DayOfWeek == DayOfWeek.Saturday)
            return true;
        if (date.Month == 1 && date.Day == 1)
            return false;
        return true;
    }
}

public class DateRange
{
    public DateTime StartDate { get; set; }
    public DateTime EndDate { get; set; }

    public IEnumerable<DateTime> GetRangeByMonth()
    {
        var currDate = DateTime.Parse(StartDate.ToString("yyyy/MM/01")).Date;
        while (currDate <= EndDate)
        {
            yield return currDate;
            currDate = currDate.AddMonths(1);
        }
    }

    public IEnumerable<DateTime> GetRangeByDay()
    {
        var currDate = StartDate;
        while (currDate <= EndDate)
        {
            if (currDate.DayOfWeek != DayOfWeek.Saturday && currDate.DayOfWeek != DayOfWeek.Sunday)
            {
                yield return currDate;
            }

            currDate = currDate.AddDays(1);
        }
    }

    public override string ToString()
    {
        return $"{StartDate.ToDateString()}~{EndDate.ToDateString()}";
    }
}