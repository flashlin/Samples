namespace SqliteCli.Helpers
{
	public class QueryableFilter
	{
		public string Name { get; set; }
		public object Value { get; set; }
		public QueryableFilterCompareEnum? Compare { get; set; }
	}
}
