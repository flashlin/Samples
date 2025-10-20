using System;

namespace Generated
{
    public class BonusWalletStatementEntity
    {
        public long TransId { get; set; }
        public int CustomerId { get; set; }
        public required string UserName { get; set; }
        public DateTime TransDate { get; set; }
        public DateTime WinLostDate { get; set; }
        public int AgtId { get; set; }
        public int MaId { get; set; }
        public int SmaId { get; set; }
        public long BonusWalletId { get; set; }
        public required string Remark { get; set; }
        public required string Description { get; set; }
        public required string CreatorName { get; set; }
        public decimal CashIn { get; set; }
        public decimal CashOut { get; set; }
        public long? StatementRefNo { get; set; }
        public DateTime? LastModifiedOn { get; set; }
    }
}
