using Microsoft.EntityFrameworkCore;
using SqliteCli.Repos;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Threading.Tasks;

namespace SqliteCli.Entities
{
	[Table("StockMap")]
	public class StockEntity
	{
		[Key]
		[StringLength(20)]
		public string Id { get; set; }

		[StringLength(100)]
		public string StockName { get; set; }
		public string StockType { get; set; }
		public decimal HandlingFee { get; set; }
	}

	//[Keyless]
	[Table("StockHistory")]
	public class StockHistoryEntity : IEquatable<StockHistoryEntity>
	{
		[Key]
		public DateTime TranDate { get; set; }
		[Key]
		public string StockId { get; set; }

		public long TradeVolume { get; set; }
		public decimal DollorVolume { get; set; }
		public decimal OpeningPrice { get; set; }
		public decimal ClosingPrice { get; set; }
		public decimal HighestPrice { get; set; }
		public decimal LowestPrice { get; set; }
		public long TransactionCount { get; set; }

		public int CompareTo(object? obj)
		{
			if (obj is StockHistoryEntity other)
			{
				if( TranDate.ToDate()== other.TranDate.ToDate()	&& StockId==other.StockId)
				{
					return 0;
				}
				if (TranDate.ToDate() > other.TranDate.ToDate())
				{
					return 1;
				}
				if (TranDate.ToDate() < other.TranDate.ToDate())
				{
					return -1;
				}
				return StockId.CompareTo(other.StockId);
			}
			return 0;
		}

		public bool Equals(StockHistoryEntity? other)
		{
			return CompareTo(other) == 1;
		}
	}
}
