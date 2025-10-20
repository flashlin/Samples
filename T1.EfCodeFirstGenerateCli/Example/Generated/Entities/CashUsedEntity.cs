using System;

namespace Generated
{
    public class CashUsedEntity
    {
        public int custid { get; set; }
        public int recommend { get; set; }
        public int mrecommend { get; set; }
        public int srecommend { get; set; }
        public decimal? CashUsed { get; set; }
        public decimal? AgtCashUsed { get; set; }
        public decimal? MaCashUsed { get; set; }
        public decimal? SmaCashUsed { get; set; }
        public decimal? RBCashUsed { get; set; }
        public decimal? RBAgtCashUsed { get; set; }
        public decimal? RBMaCashUsed { get; set; }
        public decimal? RBSmaCashUsed { get; set; }
        public decimal? GMCashUsed { get; set; }
        public decimal? GMAgtCashUsed { get; set; }
        public decimal? GMMaCashUsed { get; set; }
        public decimal? GMSmaCashUsed { get; set; }
        public required string ServiceProvider { get; set; }
        public DateTime? LastOrderOn { get; set; }
        public required string UserName { get; set; }
        public decimal? RToteCashUsed { get; set; }
        public decimal? RToteAgtCashUsed { get; set; }
        public decimal? RToteMaCashUsed { get; set; }
        public decimal? RToteSmaCashUsed { get; set; }
        public bool? IsOutstanding { get; set; }
        public decimal? LCCashUsed { get; set; }
        public decimal? LCAgtCashUsed { get; set; }
        public decimal? LCMACashUsed { get; set; }
        public decimal? LCSMACashUsed { get; set; }
        public decimal? GMStakeLimit { get; set; }
        public DateTime? GMLimitExpiredDate { get; set; }
        public decimal? GMUsedStakeLimit { get; set; }
        public DateTime? tstamp { get; set; }
        public decimal? SBStakeLimit { get; set; }
        public DateTime? SBLimitExpiredDate { get; set; }
        public decimal? SBUsedStakeLimit { get; set; }
    }
}
