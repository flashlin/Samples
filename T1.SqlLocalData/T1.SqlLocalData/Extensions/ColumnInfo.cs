namespace T1.SqlLocalData.Extensions
{
	public class ColumnInfo
	{
		public string Name { get; set; }
		public string DataType { get; set; }
		public bool IsKey { get; internal set; }
	}
}