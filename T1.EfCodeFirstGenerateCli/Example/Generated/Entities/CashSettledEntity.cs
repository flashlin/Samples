using System;

namespace Generated
{
    public class CashSettledEntity
    {
        public int custid { get; set; }
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
        public decimal? RBCashSettled { get; set; }
        public decimal? RBCashReturn { get; set; }
        public decimal? RBAgtCashSettled { get; set; }
        public decimal? RBAgtCashReturn { get; set; }
        public decimal? RBMaCashSettled { get; set; }
        public decimal? RBMaCashReturn { get; set; }
        public decimal? RBSmaCashSettled { get; set; }
        public decimal? RBSmaCashReturn { get; set; }
        public decimal? GMCashSettled { get; set; }
        public decimal? GMCashReturn { get; set; }
        public decimal? GMAgtCashSettled { get; set; }
        public decimal? GMAgtCashReturn { get; set; }
        public decimal? GMMaCashSettled { get; set; }
        public decimal? GMMaCashReturn { get; set; }
        public decimal? GMSmaCashSettled { get; set; }
        public decimal? GMSmaCashReturn { get; set; }
        public decimal? TransferIn { get; set; }
        public decimal? TransferOut { get; set; }
        public required string ServiceProvider { get; set; }
        public DateTime? LastTransferOn { get; set; }
        public int? CurrencyId { get; set; }
        public required string UserName { get; set; }
        public decimal? RToteCashSettled { get; set; }
        public decimal? RToteCashReturn { get; set; }
        public decimal? RToteAgtCashSettled { get; set; }
        public decimal? RToteAgtCashReturn { get; set; }
        public decimal? RToteMaCashSettled { get; set; }
        public decimal? RToteMaCashReturn { get; set; }
        public decimal? RToteSmaCashSettled { get; set; }
        public decimal? RToteSmaCashReturn { get; set; }
        public decimal? LCCashSettled { get; set; }
        public decimal? LCCashReturn { get; set; }
        public decimal? LCAgtCashSettled { get; set; }
        public decimal? LCMACashSettled { get; set; }
        public decimal? LCSMACashSettled { get; set; }
        public DateTime? tstamp { get; set; }
    }
}
