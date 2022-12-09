namespace CloneDatabase;

public record CloneTableReq
{
    public string SourceDbName { get; init; }
    public string SourceTableName { get; init; }
    public string TargetDbName { get; init; }
}