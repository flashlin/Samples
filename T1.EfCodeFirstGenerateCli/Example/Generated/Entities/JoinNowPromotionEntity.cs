using System;

namespace Generated
{
    public class JoinNowPromotionEntity
    {
        public int ID { get; set; }
        public int CustID { get; set; }
        public required string UserName { get; set; }
        public required string ISOCurrency { get; set; }
        public byte PromotionType { get; set; }
        public decimal? TargetTurnOver { get; set; }
        public decimal? TurnOver14 { get; set; }
        public decimal? WinLoss14 { get; set; }
        public decimal BonusAmount { get; set; }
        public DateTime ExpiryDate { get; set; }
        public byte PromotionStatus { get; set; }
        public required string Remark { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public required string PromotionCode { get; set; }
        public int? MaxEntitlement { get; set; }
        public decimal? EntitlementRate { get; set; }
        public byte? LiveIndicator { get; set; }
        public required string supportedmarkettype { get; set; }
        public decimal? AdminFeeRate { get; set; }
        public bool? isHitTarget { get; set; }
        public long? FirstBetTransId { get; set; }
        public DateTime? CreditedOn { get; set; }
    }
}
