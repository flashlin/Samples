using System;

namespace Generated
{
    public class StatementEntity
    {
        public long Transid { get; set; }
        public long Refno { get; set; }
        public int CustID { get; set; }
        public required string UserName { get; set; }
        public DateTime TransDate { get; set; }
        public DateTime WinLostDate { get; set; }
        public DateTime? CheckTime { get; set; }
        public int AgtID { get; set; }
        public int MaID { get; set; }
        public int SmaID { get; set; }
        public int TargetCustID { get; set; }
        public byte StatementType { get; set; }
        public required string Currency { get; set; }
        public int CustomerStatus { get; set; }
        public int ProductType { get; set; }
        public int? StatementStatus { get; set; }
        public required string IP { get; set; }
        public required string TransDesc { get; set; }
        public required string Remark { get; set; }
        public required string CreatorName { get; set; }
        public decimal CashIn { get; set; }
        public decimal CashOut { get; set; }
        public decimal CommIn { get; set; }
        public decimal CommOut { get; set; }
        public decimal DiscountIn { get; set; }
        public decimal DiscountOut { get; set; }
        public required string ExternalRefno { get; set; }
        public decimal? ActualRate { get; set; }
        public decimal? CasinoCashIn { get; set; }
        public decimal? CasinoCashOut { get; set; }
        public bool? IsAdjusted { get; set; }
        public DateTime? tstamp { get; set; }
        public int? TargetParentGroupId { get; set; }
    }
}
