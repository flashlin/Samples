using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Permutation
{
	public class Permutate
	{
		public List<string> Compute(string str, int n)
		{
			var list = new List<string>();

			for (int k = 0; k < str.Length; k++)
			{
				var ch = str.Substring(n, 1);
				var s = str.Substring(0, n) + str.Substring(n + 1);
				for (var i = 0; i < s.Length; i++)
				{
					s = Rotate(s);
					list.Add(ch + s);
				}
				n++;
			}
			return list;
		}


		public List<string> Compute2(string str)
		{
			var list = new List<string>();
			if (str.Length == 0)
			{
				return list;
			}

			if (str.Length == 2)
			{
				list.Add(str);
				list.Add(str.Substring(1) + str.Substring(0, 1));
				return list;
			}

			var ch = str.Substring(0, 1);
			var tail = str.Substring(1);
			foreach (var item in Compute2(tail))
			{
				list.Add(ch + tail);	
			}

			int n = 0;
			for (int k = 0; k < str.Length; k++)
			{
				str = Swap(str, pos, k);
				var child = Compute2(str, pos + 1);
				foreach (var item in child)
				{
					list.Add(item);
				}
				str = Swap(str, pos, k);
			}

			return list;
		}

		private string Swap(string str, int n, int m)
		{
			var sb = new StringBuilder();
			var ch1 = str[n];
			var ch2 = str[m];
			for (int i = 0; i < str.Length; i++)
			{
				if (i == n)
				{
					sb.Append(ch2);
					continue;
				}

				if (i == m)
				{
					sb.Append(ch1);
				}

				sb.Append(str[i]);
			}

			return sb.ToString();
		}


		public string Rotate(string str)
		{
			var ch = str.Substring(0, 1);
			return str.Substring(1) + ch;
		}



	}
}
