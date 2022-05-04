namespace SqliteCli.Repos
{
    public class ReportTransReq
    {
        public DateTime? StartDate { get; set; }
        public DateTime? EndDate { get; set; }
        public string? StockId { get; set; }
    }
}