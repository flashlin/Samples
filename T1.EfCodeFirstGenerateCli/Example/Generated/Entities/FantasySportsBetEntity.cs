using System;

namespace Generated
{
    public class FantasySportsBetEntity
    {
        public long BetId { get; set; }
        public required string RefNo { get; set; }
        public decimal Stake { get; set; }
        public int CustId { get; set; }
        public DateTime BetTime { get; set; }
        public required string Status { get; set; }
        public int? BetStatus { get; set; }
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
        public DateTime? WinLostDate { get; set; }
        public decimal? Winlost { get; set; }
        public decimal? AgtWinlost { get; set; }
        public decimal? MaWinlost { get; set; }
        public decimal? SmaWinlost { get; set; }
        public decimal? PlayerComm { get; set; }
        public decimal? AgtComm { get; set; }
        public decimal? MaComm { get; set; }
        public decimal? SmaComm { get; set; }
        public byte currency { get; set; }
        public required string SboCurrency { get; set; }
        public required string AccountId { get; set; }
        public DateTime? SettlementTime { get; set; }
        public DateTime ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public decimal CommissionableStake { get; set; }
        public bool IsResettle { get; set; }
    }
}
