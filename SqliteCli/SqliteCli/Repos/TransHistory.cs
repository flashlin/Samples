using System;
using System.ComponentModel.DataAnnotations;
using System.Text;
using T1.Standard.Common;
using T1.Standard.Extensions;

namespace SqliteCli.Repos
{
	public class TransHistory
	{
		[StringLength(10)]
		public DateTime TranTime { get; set; }

		[StringLength(7)]
		public string TranType { get; set; }

		[StringLength(9)]
		public string StockId { set; get; }

		[StringLength(30)]
		public string StockName { set; get; }

		[StringLength(6)]
		public decimal StockPrice { get; set; }

		[StringLength(10)]
		public int NumberOfShare { get; set; }

		[StringLength(5)]
		public decimal HandlingFee { get; set; }

		[StringLength(20)]
		public decimal Balance { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{TranTime.ToString("yyyy/MM/dd")}");
			sb.Append($" {TranType.ToFixLenString(7)}");
			sb.Append($" {StockId.ToFixLenString(9)}");
			sb.Append($" {StockName.ToFixLenString(30)}");
			sb.Append($" {StockPrice.ToNumberString(6)}");
			sb.Append($" {NumberOfShare.ToString().ToFixLenString(10)}");
			sb.Append($" {HandlingFee.ToNumberString(5)}");
			sb.Append($" {Balance.ToNumberString(20)}");
			return sb.ToString();
		}
	}

	public static class NumberExtension
	{
		public static string ToNumberString(this decimal number, int len)
		{
			return number.ToString("###,###,##0.00").ToFixLenString(len, AlignType.Right);
		}
	}

	public static class StringExtension
	{
		static StringExtension()
		{
			Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
		}

		public static int GetLength(this string text)
		{
			var big5 = Encoding.GetEncoding(950);
			return big5.GetBytes(text).Length;
			//int myStringCount = 0;
			//for (int i = 0; i < text.Length; i++)
			//{
			//	byte[] tIntByte = Encoding.UTF8.GetBytes(text.Substring(i, 1));
			//	if (tIntByte.Length > 1)
			//	{
			//		myStringCount += 2;
			//	}
			//	else
			//	{
			//		myStringCount++;
			//	}
			//}
			//return myStringCount;
		}

		public static string ToFixLenString(this string text, int len, AlignType align = AlignType.Left)
		{
			if (string.IsNullOrEmpty(text))
			{
				return new string(' ', len);
			}

			var textLen = text.GetLength();

			if (textLen > len)
			{
				var big5 = Encoding.GetEncoding(950);
				var textBuff = big5.GetBytes(text);
				var buff = new byte[len];
				Array.Copy(textBuff, buff, len);
				var str = big5.GetString(buff);
				return str;
			}


			var spaces = new string(' ', len - textLen);
			switch (align)
			{
				case AlignType.Left:
					return $"{text}{spaces}";
				default:
					return $"{spaces}{text}";
			}
		}
	}

	public enum AlignType
	{
		Left,
		Right
	}
}
