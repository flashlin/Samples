namespace SqliteCli.Helpers;

public class DateRange
{
    public DateTime StartDate { get; set; }
    public DateTime EndDate { get; set; }

    public IEnumerable<DateTime> GetRangeByMonth()
    {
        var currDate = DateTime.Parse(StartDate.ToString("yyyy/MM/01")).Date;
        while(currDate <= EndDate)
        {
            yield return currDate;
            currDate = currDate.AddMonths(1);
        }
    }
}