using System.Text;

namespace SqliteCli.Entities
{
	public class TransHistory
	{
		public DateTime TranTime { get; set; }
		public string TranType { get; set; }
		public string StockId { set; get; }
		public string StockName { set; get; }
		public decimal StockPrice { get; set; }
		public int NumberOfShare { get; set; }
		public decimal HandlingFee { get; set; }
		public decimal Balance { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{TranTime.ToString("yyyy/MM/dd")}");
			sb.Append($" {TranType}");
			sb.Append($" {StockId}");
			sb.Append($" {StockName}");
			sb.Append($" {StockPrice}");
			sb.Append($" {NumberOfShare}");
			sb.Append($" {HandlingFee}");
			sb.Append($" {Balance}");
			return sb.ToString();
		}
	}

}
