using System.Text;
using T1.Standard.Extensions;
using T1.Standard.DynamicCode;
using System.Reflection;

namespace SqliteCli.Repos
{
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
