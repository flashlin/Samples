namespace SqliteCli.Repos
{
	[AttributeUsage(AttributeTargets.Property, AllowMultiple = false)]
	public class DecimalStringAttribute : DisplayStringAttribute
	{
		public DecimalStringAttribute(int maxLength) 
			: base("###,###,##0.00", maxLength, AlignType.Right)
		{
		}
	}
}
