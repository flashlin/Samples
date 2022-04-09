namespace PreviewLibrary.Pratt.TSql
{
	public enum Precedence
	{
		ASSIGNMENT = 1,
		CONDITIONAL,
		SUM,
		PRODUCT,
		EXPONENT,
		PREFIX,
		POSTFIX,
		CALL,
	}
}
