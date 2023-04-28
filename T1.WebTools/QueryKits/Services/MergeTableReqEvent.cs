namespace QueryKits.Services;

public class MergeTableReqEvent : EventArgs
{
    public string LeftTableName { get; set; }
    public string RightTableName { get; set; }
}