using System;

namespace Generated
{
    public class MpCashUsedEntity
    {
        public int CustomerId { get; set; }
        public int recommend { get; set; }
        public int mrecommend { get; set; }
        public int srecommend { get; set; }
        public decimal? CashUsed { get; set; }
        public decimal? AgtCashUsed { get; set; }
        public decimal? MaCashUsed { get; set; }
        public decimal? SmaCashUsed { get; set; }
        public required string ServiceProvider { get; set; }
        public DateTime? LastOrderOn { get; set; }
        public required string Username { get; set; }
        public bool? IsOutstanding { get; set; }
        public DateTime? tstamp { get; set; }
        public decimal? SBStakeLimit { get; set; }
        public DateTime? SBLimitExpiredDate { get; set; }
        public decimal? SBUsedStakeLimit { get; set; }
    }
}
