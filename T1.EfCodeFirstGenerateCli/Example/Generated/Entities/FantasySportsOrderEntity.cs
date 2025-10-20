using System;

namespace Generated
{
    public class FantasySportsOrderEntity
    {
        public long OrderId { get; set; }
        public required string RefNo { get; set; }
        public decimal Stake { get; set; }
        public int CustId { get; set; }
        public DateTime BetTime { get; set; }
        public DateTime CreatedTime { get; set; }
        public decimal? ActualRate { get; set; }
        public int? AgtId { get; set; }
        public int? MaId { get; set; }
        public int? SmaId { get; set; }
        public decimal? AgtPT { get; set; }
        public decimal? MaPT { get; set; }
        public decimal? SmaPT { get; set; }
        public decimal? PlayerCommRate { get; set; }
        public decimal? AgtCommRate { get; set; }
        public decimal? MaCommRate { get; set; }
        public decimal? SmaCommRate { get; set; }
        public int? DirectCustId { get; set; }
        public int MemberStatus { get; set; }
        public byte currency { get; set; }
        public required string SboCurrency { get; set; }
        public required string AccountId { get; set; }
        public DateTime ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
    }
}
