namespace T1.SqlSharp.Expressions;

public enum SelectItemType
{
    /// <summary>
    /// Simple column or expression
    /// </summary>
    Column,

    /// <summary>
    /// Nested subquery
    /// </summary>
    SubQuery
}