using System;
using System.ComponentModel.DataAnnotations;
using System.Text;
using T1.Standard.Common;
using T1.Standard.Extensions;
using T1.Standard.DynamicCode;
using System.Reflection;

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

	public class ReportTranItem
	{
		[StringFixed(7)]
		public string TranType { get; set; }

		[StringFixed(9)]
		public string StockId { set; get; }

		[StringFixed(30)]
		public string StockName { set; get; }

		//2022-04-04
		//假如這邊 property 用 decimal or double
		//卻噴出 ERROR: Sqlite AVG(xxx) 出來是 double, but dapper 卻 parse to int64
		//只好改為 string
		[StringFixed(6)]
		public string MinStockPrice { get; set; }

		[StringFixed(6)]
		public string AvgStockPrice { get; set; }

		[StringFixed(6)]
		public string MaxStockPrice { get; set; }

		[StringFixed(7)]
		public int NumberOfShare { get; set; }

		[DecimalString(7)]
		public decimal HandlingFee { get; set; }

		[DecimalString(20)]
		public decimal Balance { get; set; }
	}


	public interface IDisplayString
	{
		string ToDisplayString(object value);
	}

	//[AttributeUsage(AttributeTargets.Property, AllowMultiple = false)]
	//public class StringFixedAttribute : Attribute, IDisplayString
	//{
	//	public StringFixedAttribute(int maxLength, AlignType alignType = AlignType.Left)
	//	{
	//		MaxLength = maxLength;
	//		AlignType = alignType;
	//	}

	//	public int MaxLength { get; }
	//	public AlignType AlignType { get; }

	//	public string ToDisplayString(object value)
	//	{
	//		return $"{value}".ToFixLenString(MaxLength, AlignType);
	//	}
	//}

	[AttributeUsage(AttributeTargets.Property, AllowMultiple = false)]
	public class StringFixedAttribute : Attribute, IDisplayString
	{
		public StringFixedAttribute(int maxLength, AlignType alignType = AlignType.Left)
		{
			MaxLength = maxLength;
			AlignType = alignType;
		}

		public int MaxLength { get; }
		public AlignType AlignType { get; }

		public string ToDisplayString(object value)
		{
			return $"{value}".ToFixLenString(MaxLength, AlignType);
		}
	}

	[AttributeUsage(AttributeTargets.Property, AllowMultiple = false)]
	public class DecimalStringAttribute : Attribute, IDisplayString
	{
		public DecimalStringAttribute(int maxLength)
		{
			MaxLength = maxLength;
		}

		public int MaxLength { get; private set; }

		public string ToDisplayString(object value)
		{
			var number = (decimal)value;
			return number.ToString("###,###,##0.00").ToFixLenString(MaxLength, AlignType.Right);
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

		public static string GetDisplayTitle(this object obj)
		{
			var sb = new StringBuilder();
			var delimiter = new StringBuilder();
			var clazz = ReflectionClass.Reflection(obj.GetType());
			var first = true;
			foreach (var prop in clazz.Properties.Values)
			{
				var propInfo = (PropertyInfo)prop.Info;
				var decimalAttr = propInfo.GetCustomAttribute<DecimalStringAttribute>(false);
				if (decimalAttr != null)
				{
					var value = prop.Getter(obj);
					if (!first)
					{
						sb.Append(" ");
						delimiter.Append(" ");
					}

					sb.Append(prop.Name.ToFixLenString(decimalAttr.MaxLength, AlignType.Right));
					delimiter.Append(new String('-', decimalAttr.MaxLength));
					first = false;
					continue;
				}

				var displayAttr = propInfo.GetCustomAttribute<DisplayStringAttribute>(false);
				if (displayAttr != null)
				{
					if (!first)
					{
						sb.Append(" ");
						delimiter.Append(" ");
					}
					var value = prop.Getter(obj);
					
					sb.Append(displayAttr.ToDisplayString(prop.Name));

					delimiter.Append(new String('-', displayAttr.MaxLength));
					first = false;
					continue;
				}
			}
			return sb.ToString() + "\r\n" + delimiter.ToString();
		}

		public static string GetDisplayValue(this object obj)
		{
			var sb = new StringBuilder();
			var clazz = ReflectionClass.Reflection(obj.GetType());
			var first = true;
			foreach (var prop in clazz.Properties.Values)
			{
				var propInfo = (PropertyInfo)prop.Info;
				var decimalAttr = propInfo.GetCustomAttribute<DecimalStringAttribute>(false);
				if (decimalAttr != null)
				{
					var value = (decimal)prop.Getter(obj);
					if (!first)
					{
						sb.Append(" ");
					}

					sb.Append(value.ToNumberString(decimalAttr.MaxLength));
					first = false;
					continue;
				}

				var displayAttr = propInfo.GetCustomAttribute<DisplayStringAttribute>(false);
				if (displayAttr != null)
				{
					if (!first)
					{
						sb.Append(" ");
					}
					var value = prop.Getter(obj);
					sb.Append(displayAttr.ToDisplayString(value));
					first = false;
					continue;
				}
			}
			return sb.ToString();
		}

		public static int GetLength(this string text)
		{
			var big5 = Encoding.GetEncoding(950);
			return big5.GetBytes(text).Length;
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
}
