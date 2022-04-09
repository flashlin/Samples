namespace SqliteCli.Repos
{
	public class StockExchangeData
	{
		[DisplayString("", 10)]
		public DateTime Date { get; set; }
		
		[DisplayString("", 7)]
		public string StockId { get; set; }
		
		[DisplayString("", 7)]
		public long TradeVolume { get; set; }

		[DisplayString("", 7)]
		public decimal DollorVolume { get; set; }
		
		[DisplayString("", 7)]
		public decimal OpeningPrice { get; set; }

		[DisplayString("", 7)]
		public decimal HighestPrice { get; set; }

		[DisplayString("", 7)]
		public decimal LowestPrice { get; set; }
		
		[DisplayString("", 7)]
		public decimal ClosingPrice { get; set; }
		
		[DisplayString("", 7)]
		public decimal Change { get; set; }

		[DisplayString("", 7)]
		public long Transaction { get; set; }
	}
}
