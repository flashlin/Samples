using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SqliteCli.Repos
{
	public static class DumpExtension
	{
		public static void Dump(this List<TransHistory> result)
		{
			if (result.Count > 0)
			{
				var title = result.First().GetDisplayTitle();
				Console.WriteLine(title);
			}
			foreach (var item in result)
			{
				Console.WriteLine(item.GetDisplayValue());
			}
			if (result.Count > 0)
			{
				var summary = new TransHistory
				{
					TranTime = DateTime.Now,
					TranType = "Summary",
					NumberOfShare = result.Sum(x => x.NumberOfShare),
					HandlingFee = result.Sum(x => x.HandlingFee),
					Balance = result.Sum(x => x.Balance),
				};
				Console.WriteLine(summary.GetDisplayValue());
			}
		}

		public static void Dump<T>(this List<T> result)
			where T : class, new()
		{
			result.DumpList();
			if (result.Count > 0)
			{
				var summary = result.ToSummary<T>();
				var line = summary.GetDisplayValue();
				Console.WriteLine(line);
			}
		}

		public static void DumpList<T>(this List<T> result)
			where T : class, new()
		{
			if (result.Count > 0)
			{
				var title = result[0].GetDisplayTitle();
				Console.WriteLine(title);
			}
			foreach (var item in result)
			{
				Console.WriteLine(item.GetDisplayValue());
			}
		}
	}
}
