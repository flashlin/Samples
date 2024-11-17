namespace SqlSharpLit.Common.ParserLit;

public class SqlConstraint
{
    public string ConstraintName { get; set; } = string.Empty;
    public string ConstraintType { get; set; } = string.Empty;
    public string Clustered { get; set; } = string.Empty;
    public List<SqlColumnConstraint> Columns { get; set; } = [];
    public List<SqlWithToggle> WithToggles { get; set; } = [];
    public string On { get; set; } = string.Empty;
}