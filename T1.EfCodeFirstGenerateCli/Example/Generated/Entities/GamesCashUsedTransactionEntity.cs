using System;

namespace Generated
{
    public class GamesCashUsedTransactionEntity
    {
        public required string TransactionId { get; set; }
        public required string ExternalRefNo { get; set; }
        public int CustomerId { get; set; }
        public decimal Amount { get; set; }
        public DateTime CreatedOn { get; set; }
        public bool IsVerified { get; set; }
        public decimal AgtPT { get; set; }
        public decimal MaPT { get; set; }
        public decimal SmaPT { get; set; }
        public decimal? BettingBudgetAmount { get; set; }
    }
}
