namespace PreviewLibrary
{
	public class SetIdentityInsertExpr : SqlExpr
	{
		public bool Toggle { get; set; }
		public override string ToString()
		{
			var onOff = Toggle ? "ON" : "OFF";
			return $"SET IDENTITY_INSERT {onOff}";
		}
	}
}