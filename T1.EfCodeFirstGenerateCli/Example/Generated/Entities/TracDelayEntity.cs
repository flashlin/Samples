using System;

namespace Generated
{
    public class TracDelayEntity
    {
        public int Id { get; set; }
        public required string FromAccountId { get; set; }
        public int FromCustomerID { get; set; }
        public DateTime? CreateDate { get; set; }
        public DateTime? WinlostDate { get; set; }
        public required string ToAccountId { get; set; }
        public int ToCustomerId { get; set; }
        public decimal Amount { get; set; }
        public decimal? ExchangeRate { get; set; }
        public required string Description { get; set; }
        public required string Remark { get; set; }
        public int? Status { get; set; }
        public required string UpdatedBy { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? UpdatedOn { get; set; }
        public decimal? FromBankFee { get; set; }
        public decimal? ToBankFee { get; set; }
        public bool? IsDeductedFromSender { get; set; }
        public bool IsAutoInsertCurrencyTrac { get; set; }
        public required string CurrencyTracDescription { get; set; }
    }
}
