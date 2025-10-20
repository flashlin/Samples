using System;

namespace Generated
{
    public class BonusWalletDailyStatementEntity
    {
        public long TransId { get; set; }
        public int CustomerId { get; set; }
        public long BonusWalletId { get; set; }
        public required string UserName { get; set; }
        public DateTime WinLostDate { get; set; }
        public int AgtId { get; set; }
        public int MaId { get; set; }
        public int SmaId { get; set; }
        public byte StatementType { get; set; }
        public int ProductType { get; set; }
        public decimal CashIn { get; set; }
        public decimal CashOut { get; set; }
        public decimal TotalCashIn { get; set; }
        public decimal TotalCashOut { get; set; }
        public DateTime TransDate { get; set; }
        public DateTime? LastModifiedOn { get; set; }
    }
}
