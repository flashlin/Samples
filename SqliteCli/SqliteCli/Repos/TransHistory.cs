using System;
using System.ComponentModel.DataAnnotations;
using T1.Standard.Common;

namespace SqliteCli.Repos
{
	public class TransHistory
	{
		[DisplayString("", 5)]
		public long Id { get; set; }

		[DisplayString("yyyy/MM/dd", 10)]
		public DateTime TranTime { get; set; }

		[DisplayString("", 7)]
		public string TranType { get; set; }

		[DisplayString("", 9)]
		public string StockId { set; get; }

		[DisplayString("", 30)]
		public string StockName { set; get; }

		[DecimalString(6)]
		public decimal StockPrice { get; set; }

		[DisplayString("", 7, AlignType.Right)]
		public int NumberOfShare { get; set; }

		[DecimalString(7)]
		public decimal HandlingFee { get; set; }

		[DecimalString(20)]
		public decimal Balance { get; set; }
	}
}
