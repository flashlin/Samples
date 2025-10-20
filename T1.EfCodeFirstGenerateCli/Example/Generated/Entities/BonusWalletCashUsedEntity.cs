using System;

namespace Generated
{
    public class BonusWalletCashUsedEntity
    {
        public long BonusWalletId { get; set; }
        public int CustomerId { get; set; }
        public required string UserName { get; set; }
        public int Recommend { get; set; }
        public int Mrecommend { get; set; }
        public int Srecommend { get; set; }
        public decimal? CashUsed { get; set; }
        public decimal? AgtCashUsed { get; set; }
        public decimal? MaCashUsed { get; set; }
        public decimal? SmaCashUsed { get; set; }
        public DateTime? LastOrderOn { get; set; }
        public DateTime? LastModifiedOn { get; set; }
    }
}
