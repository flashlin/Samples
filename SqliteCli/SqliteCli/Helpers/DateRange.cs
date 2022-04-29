using SqliteCli.Repos;

namespace SqliteCli.Helpers;

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