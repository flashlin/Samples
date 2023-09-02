namespace T1.ParserKit.Core.Utilities
{
    public static class AssertionExtensions
    {
        #region AssertNotNull

        public static void AssertNotNull<T>(this T instance, string message = "Expected a non-null object reference.")
            where T : class
        {
            if (instance == null)
                throw new NullReferenceException(message);
        }

        #endregion //AssertNotNull

        #region AssertNotNegative

        public static void AssertNotNegative(this int value, string message = "The specified value cannot be less than zero.")
        {
            if (value < 0)
                throw new InvalidOperationException(message);
        }

        #endregion //AssertNotNegative
    }
}
