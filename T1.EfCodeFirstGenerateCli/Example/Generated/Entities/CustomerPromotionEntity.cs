using System;

namespace Generated
{
    public class CustomerPromotionEntity
    {
        public int VoucherId { get; set; }
        public int CustId { get; set; }
        public int PromotionType { get; set; }
        public bool? IsEnabled { get; set; }
        public DateTime? EffectiveDate { get; set; }
        public DateTime? ExpiryDate { get; set; }
        public decimal? CashEntitled { get; set; }
        public decimal? CashUsed { get; set; }
        public required string Detail { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public decimal? CashEntitledInSGD { get; set; }
        public bool IsRead { get; set; }
    }
}
