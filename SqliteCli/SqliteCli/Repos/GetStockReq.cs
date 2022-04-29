namespace SqliteCli.Repos
{
	public class GetStockReq
	{
		public DateTime StartDate { get; set; }
		public DateTime EndDate { get; set; }
		public string StockId { get; set; }
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
