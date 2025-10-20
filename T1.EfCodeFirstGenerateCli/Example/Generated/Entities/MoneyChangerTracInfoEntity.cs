using System;

namespace Generated
{
    public class MoneyChangerTracInfoEntity
    {
        public int Id { get; set; }
        public int TracDelayId { get; set; }
        public required string InvoiceNumber { get; set; }
        public required string Jurisdiction { get; set; }
        public int Status { get; set; }
        public short Batch { get; set; }
        public int MoneyChangerSenderInfoId { get; set; }
        public int BankInfoId { get; set; }
        public decimal? BankReceivedAmount { get; set; }
        public decimal? BankFee { get; set; }
        public required string SlipDetails { get; set; }
        public required string Currency { get; set; }
        public DateTime? DealingDate { get; set; }
        public DateTime? SlipDate { get; set; }
        public DateTime? BankReceivedDate { get; set; }
        public DateTime? UpdatedOn { get; set; }
        public required string UpdatedBy { get; set; }
        public required string Note { get; set; }
        public int? MoneyChangerTargetSenderInfoId { get; set; }
        public int? PartnershipTargetSenderInfoId { get; set; }
        public required string TargetSenderType { get; set; }
    }
}
