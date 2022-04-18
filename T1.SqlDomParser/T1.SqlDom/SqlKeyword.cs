namespace T1.SqlDomParser
{
	struct SqlKeyword
	{
		public string Text { get; }
		public SqlToken Token { get; }

		public SqlKeyword(string text, SqlToken token)
		{
			if (text == null) throw new ArgumentNullException(nameof(text));
			Text = text;
			Token = token;
		}
	}
}
