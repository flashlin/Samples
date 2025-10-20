using System;

namespace Generated
{
    public class BonusWalletCashSettledEntity
    {
        public long BonusWalletId { get; set; }
        public int CustomerId { get; set; }
        public required string UserName { get; set; }
        public int Recommend { get; set; }
        public int Mrecommend { get; set; }
        public int Srecommend { get; set; }
        public decimal? CashSettled { get; set; }
        public decimal? CashReturn { get; set; }
        public decimal? AgtCashSettled { get; set; }
        public decimal? AgtCashReturn { get; set; }
        public decimal? MaCashSettled { get; set; }
        public decimal? MaCashReturn { get; set; }
        public decimal? SmaCashSettled { get; set; }
        public decimal? SmaCashReturn { get; set; }
        public decimal? TransferIn { get; set; }
        public decimal? TransferOut { get; set; }
        public DateTime? LastTransferOn { get; set; }
        public DateTime? LastModifiedOn { get; set; }
    }
}
