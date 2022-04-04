using System.ComponentModel.DataAnnotations;

namespace SqliteCli.Entities
{
	public class TransEntity
	{
		[Key]
		public long Id { get; set; }
		public DateTime TranTime { get; set; }
		public string TranType { get; set; }
		public string StockId { get; set; }
		public decimal StockPrice { get; set; }
		public int NumberOfShare { get; set; }
		public decimal HandlingFee { get; set; }
		public decimal Balance { get; set; }
	}

}
