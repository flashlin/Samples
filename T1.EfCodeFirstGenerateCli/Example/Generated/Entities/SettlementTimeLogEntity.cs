using System;

namespace Generated
{
    public class SettlementTimeLogEntity
    {
        public int MatchResultId { get; set; }
        public required string SPName { get; set; }
        public int BetCount { get; set; }
        public DateTime? StartTime { get; set; }
        public DateTime? AfterBetTrans { get; set; }
        public DateTime? AfterCashSettled { get; set; }
        public DateTime? AfterDailyStatement { get; set; }
        public DateTime? AfterSettledBetTrans { get; set; }
    }
}
