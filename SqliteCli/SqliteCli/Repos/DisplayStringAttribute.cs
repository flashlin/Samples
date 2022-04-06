namespace SqliteCli.Repos
{
	[AttributeUsage(AttributeTargets.Property, AllowMultiple = false)]
	public class DisplayStringAttribute : Attribute, IDisplayString
	{
		public DisplayStringAttribute(string formatString, int maxLength, AlignType alignType = AlignType.Left)
		{
			FormatString = formatString;
			MaxLength = maxLength;
			AlignType = alignType;
		}

		public string FormatString { get; }
		public int MaxLength { get; }
		public AlignType AlignType { get; }

		public string ToDisplayString(object value)
		{
			if (string.IsNullOrEmpty(FormatString))
			{
				return $"{value}".ToFixLenString(MaxLength, AlignType);
			}
			return string.Format("{0:" + FormatString + "}", value).ToFixLenString(MaxLength, AlignType);
		}
	}
}
