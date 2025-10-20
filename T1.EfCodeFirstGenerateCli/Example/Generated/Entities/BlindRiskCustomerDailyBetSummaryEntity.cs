using System;

namespace Generated
{
    public class BlindRiskCustomerDailyBetSummaryEntity
    {
        public int CustId { get; set; }
        public DateTime WinlostDate { get; set; }
        public decimal Turnover { get; set; }
        public int StakeLessThan100BetCount { get; set; }
        public int TotalBetCount { get; set; }
    }
}
