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
		[DisplayString(10)]
		public DateTime TranTime { get; set; }

		[DisplayString(7)]
		public string TranType { get; set; }

		[DisplayString(9)]
		public string StockId { set; get; }

		[DisplayString(30)]
		public string StockName { set; get; }

		[DecimalString(6)]
		public decimal StockPrice { get; set; }

		[DisplayString(7)]
		public int NumberOfShare { get; set; }

		[DecimalString(7)]
		public decimal HandlingFee { get; set; }

		[DecimalString(20)]
		public decimal Balance { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{TranTime.ToString("yyyy/MM/dd")}");
			sb.Append($" {TranType.ToFixLenString(7)}");
			sb.Append($" {StockId.ToFixLenString(9)}");
			sb.Append($" {StockName.ToFixLenString(30)}");
			sb.Append($" {StockPrice.ToNumberString(6)}");
			sb.Append($" {NumberOfShare.ToString().ToFixLenString(7)}");
			sb.Append($" {HandlingFee.ToNumberString(7)}");
			sb.Append($" {Balance.ToNumberString(20)}");
			return sb.ToString();
		}
	}

	public interface IDisplayString
	{
		string ToDisplayString(object value);
	}

	[AttributeUsage(AttributeTargets.Property, AllowMultiple = false)]
	public class DisplayStringAttribute : Attribute, IDisplayString
	{
		public DisplayStringAttribute(int maxLength, AlignType alignType = AlignType.Left)
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
					if(!first)
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
					if(!first)
					{
						sb.Append(" ");
						delimiter.Append(" ");
					}
					var value = prop.Getter(obj);
					sb.Append(prop.Name.ToFixLenString(displayAttr.MaxLength));
					delimiter.Append(new String('-', displayAttr.MaxLength));
					first = false;
					continue;
				}
			}
			return sb.ToString() + "\r\n" + delimiter.ToString(); 
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

	public enum AlignType
	{
		Left,
		Right
	}
}
