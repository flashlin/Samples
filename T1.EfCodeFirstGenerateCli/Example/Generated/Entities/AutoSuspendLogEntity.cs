using System;

namespace Generated
{
    public class AutoSuspendLogEntity
    {
        public int Id { get; set; }
        public int CustomerId { get; set; }
        public required string AccountId { get; set; }
        public required string Currency { get; set; }
        public decimal? MarketRate { get; set; }
        public decimal? TotalBalanceInBaseCurrency { get; set; }
        public decimal? TotalOutstandingInBaseCurrency { get; set; }
        public decimal? SuspendLimit { get; set; }
        public DateTime SuspendedOn { get; set; }
        public int? GroupId { get; set; }
        public required string GroupName { get; set; }
        public required string SuspendedBy { get; set; }
    }
}
