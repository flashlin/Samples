namespace QueryKits.Services;

public class MergeTableReqEvent : EventArgs
{
    public string LeftTableName { get; set; } = string.Empty;
    public string RightTableName { get; set; } = string.Empty;
}