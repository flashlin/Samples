using System.Data;
using T1.Standard.Common;
using T1.Standard.DynamicCode;

namespace SqliteCli.Repos
{
	public class StockExchangeRawData
	{
		public static Dictionary<string, string> FieldNames = new Dictionary<string, string>
		{
			{ "日期", nameof(StockExchangeData.Date) },
			{ "成交股數", nameof(StockExchangeData.TradeVolume) },
			{ "成交金額", nameof(StockExchangeData.DollorVolume) },
			{ "開盤價", nameof(StockExchangeData.OpeningPrice) },
			{ "最高價", nameof(StockExchangeData.HighestPrice) },
			{ "最低價", nameof(StockExchangeData.LowestPrice) },
			{ "收盤價", nameof(StockExchangeData.ClosingPrice) },
			{ "漲跌價差", nameof(StockExchangeData.Change) },
			{ "成交筆數", nameof(StockExchangeData.Transaction) },
		};

		public string Stat { get; set; }
		public string Date { get; set; }
		public string Title { get; set; }
		public List<string> Fields { get; set; }
		public List<List<string>> Data { get; set; }
		public List<string> Notes { get; set; }

		public IEnumerable<StockExchangeData> GetStockList(string stockId)
		{
			if( Data == null )
			{
				yield break;
			}

			var stockTranObjInfo = ReflectionClass.Reflection(typeof(StockExchangeData));
			foreach (var dataItem in Data)
			{
				var stockTran = new StockExchangeData();
				stockTran.StockId = stockId;
				foreach (var item in Fields.Select((value, idx) => new { name = FieldNames[value], idx }))
				{
					var valueStr = dataItem[item.idx];
					var propInfo = stockTranObjInfo.Properties[item.name];
					var value = (object)valueStr;
					if (propInfo.PropertyType != typeof(string))
					{
						if (propInfo.PropertyType.IsValueType)
						{
							valueStr = valueStr.Replace(",", "");
							valueStr = valueStr.Replace("X", "");
						}
						value = valueStr.ChangeType(propInfo.PropertyType);

						if( propInfo.Name == nameof(StockExchangeData.Date))
						{
							var date = (DateTime)value;
							date = date.AddYears(1911);
							value = date;
						}
					}
					propInfo.Setter(stockTran, value);
				}
				yield return stockTran;
			}
		}
	}
}
