using System;

namespace Generated
{
    public class FantasyRankingEntity
    {
        public int CustId { get; set; }
        public required string FirstName { get; set; }
        public decimal NetWinLost { get; set; }
        public int Ranking { get; set; }
        public int PreviousRanking { get; set; }
        public DateTime LastModifiedOn { get; set; }
        public required string Email { get; set; }
        public int NumberOfWinBets { get; set; }
    }
}
