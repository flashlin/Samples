namespace SqliteCli.Repos
{
	public class StockExchangeData
	{
		[DisplayString("yyyy-MM-dd", 10)]
		public DateTime Date { get; set; }
		
		[DisplayString("", 7)]
		public string StockId { get; set; }
		
		/// <summary>
		/// 成交股數
		/// </summary>
		[DisplayString("", 7)]
		public long TradeVolume { get; set; }

		/// <summary>
		/// 成交金額
		/// </summary>
		[DisplayString("", 7)]
		public long DollorVolume { get; set; }
		
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

		/// <summary>
		/// 成交筆數
		/// </summary>
		[DisplayString("", 7)]
		public long  Transaction { get; set; }
	}
}
