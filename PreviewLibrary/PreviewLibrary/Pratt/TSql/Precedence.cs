namespace PreviewLibrary.Pratt.TSql
{
	public enum Precedence
	{
		ASSIGNMENT = 1,
		CONDITIONAL,
		CONCAT,
		SUM,
		PRODUCT,
		EXPONENT,
		PREFIX,
		POSTFIX,
		CALL,
	}
}
