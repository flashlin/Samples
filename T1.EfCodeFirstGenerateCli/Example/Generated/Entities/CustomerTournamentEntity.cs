using System;

namespace Generated
{
    public class CustomerTournamentEntity
    {
        public int Id { get; set; }
        public int CustomerId { get; set; }
        public required string Currency { get; set; }
        public required string LoginName { get; set; }
        public int TournamentType { get; set; }
        public int? TournamentCategory { get; set; }
        public DateTime JoinDate { get; set; }
        public decimal? TotalWinLose { get; set; }
        public decimal? HistoryWinLose { get; set; }
        public int? TotalWinningBet { get; set; }
        public int? TotalLosingBet { get; set; }
        public int? TotalBet { get; set; }
        public int? HistoryWinningBet { get; set; }
        public int? HistoryLosingBet { get; set; }
        public int? HistoryTotalBet { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime ModifiedOn { get; set; }
    }
}
