namespace SqliteCli.Repos
{
	public class ListTransReq
	{
		public DateTime? StartTime { get; set; }
		public DateTime? EndTime { get; set; }
		public string? StockId { get; set; }
	}

}
