using System;

namespace Generated
{
    public class SettledBetTransEntity
    {
        public long TransId { get; set; }
        public int MatchResultId { get; set; }
        public required string ActionName { get; set; }
        public int ActionId { get; set; }
        public int CustId { get; set; }
        public decimal ActualStake { get; set; }
        public decimal Stake { get; set; }
        public required string WinLost { get; set; }
        public DateTime WinLostDate { get; set; }
        public required string Status { get; set; }
        public int BetStatus { get; set; }
        public byte? StatusWinlost { get; set; }
        public bool IsFreeBet { get; set; }
        public bool IsCoverBet { get; set; }
        public byte? OldStatusWinlost { get; set; }
        public required string OldWinLost { get; set; }
        public long? ID { get; set; }
    }
}
