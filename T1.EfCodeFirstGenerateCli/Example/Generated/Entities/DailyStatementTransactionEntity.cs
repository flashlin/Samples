using System;

namespace Generated
{
    public class DailyStatementTransactionEntity
    {
        public long BetReferenceId { get; set; }
        public int ProductType { get; set; }
        public long BonusWalletId { get; set; }
        public int CustomerId { get; set; }
        public decimal CashIn { get; set; }
        public decimal CashOut { get; set; }
        public DateTime CreatedOn { get; set; }
        public bool IsVerified { get; set; }
    }
}
