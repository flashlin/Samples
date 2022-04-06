namespace SqliteCli.Repos
{
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
}
