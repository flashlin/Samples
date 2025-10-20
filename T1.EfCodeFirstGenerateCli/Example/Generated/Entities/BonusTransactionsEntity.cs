using System;

namespace Generated
{
    public class BonusTransactionsEntity
    {
        public int Id { get; set; }
        public DateTime WinLostDate { get; set; }
        public decimal Amount { get; set; }
        public required string FromAccountId { get; set; }
        public required string ToAccountId { get; set; }
        public int FromCustomerId { get; set; }
        public int ToCustomerId { get; set; }
        public required string FromCurrency { get; set; }
        public required string ToCurrency { get; set; }
        public decimal FromActualRate { get; set; }
        public decimal ToActualRate { get; set; }
        public int ServiceProvider { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public required string Status { get; set; }
        public required string Remark { get; set; }
        public required string Description { get; set; }
        public bool? IsSendPersonalMessage { get; set; }
        public required string DescriptionTranslated { get; set; }
        public required string RequestIdentifier { get; set; }
        public int ProductType { get; set; }
    }
}
