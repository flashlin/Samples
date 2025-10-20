using System;

namespace Generated
{
    public class CustomerPromotionSignUpEntity
    {
        public int CustId { get; set; }
        public int PromotionType { get; set; }
        public required string Option { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public decimal? CurrentWinlost { get; set; }
        public decimal? HistoryWinlost { get; set; }
        public int? CurrentBetCount { get; set; }
        public int? HistoryBetCount { get; set; }
        public DateTime JoinDate { get; set; }
        public int PromotionCategory { get; set; }
        public required string Currency { get; set; }
        public required string LoginName { get; set; }
        public int? CurrentTotalBetCount { get; set; }
        public int? HistoryTotalBetCount { get; set; }
    }
}
