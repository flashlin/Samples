namespace PreviewLibrary.Pratt.TSql
{
	public enum Precedence
	{
		ASSIGNMENT = 1,
		CONDITIONAL,
		CONCAT,
		COMPARE,
		SUM,
		PRODUCT,
		EXPONENT,
		PREFIX,
		POSTFIX,
		CALL,
	}
}
