using System;

namespace Generated
{
    public class MpCashSettledEntity
    {
        public int CustomerId { get; set; }
        public int recommend { get; set; }
        public int mrecommend { get; set; }
        public int srecommend { get; set; }
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
        public required string ServiceProvider { get; set; }
        public DateTime? LastTransferOn { get; set; }
        public int? CurrencyId { get; set; }
        public required string Username { get; set; }
        public DateTime? tstamp { get; set; }
    }
}
