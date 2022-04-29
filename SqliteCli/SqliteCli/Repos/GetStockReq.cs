using SqliteCli.Helpers;

namespace SqliteCli.Repos
{
	public class GetStockReq
	{
		public string StockId { get; set; }
		public DateRange DateRange { get; set; }
	}

	public static class DateExtension
	{
		public static string ToDateString(this DateTime time)
		{
			return time.ToString("yyyy-MM-dd");
		}
		public static DateTime ToDate(this DateTime time)
		{
			return DateTime.Parse(time.ToDateString());
		}
	}
}
