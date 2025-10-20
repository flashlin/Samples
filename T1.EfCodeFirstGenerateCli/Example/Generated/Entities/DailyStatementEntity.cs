using System;

namespace Generated
{
    public class DailyStatementEntity
    {
        public long Transid { get; set; }
        public int CustID { get; set; }
        public required string UserName { get; set; }
        public DateTime WinLostDate { get; set; }
        public int AgtID { get; set; }
        public int MaID { get; set; }
        public int SmaID { get; set; }
        public byte StatementType { get; set; }
        public required string Currency { get; set; }
        public int CustomerStatus { get; set; }
        public int ProductType { get; set; }
        public int? StatementStatus { get; set; }
        public decimal CashIn { get; set; }
        public decimal CashOut { get; set; }
        public decimal CommIn { get; set; }
        public decimal CommOut { get; set; }
        public decimal DiscountIn { get; set; }
        public decimal DiscountOut { get; set; }
        public decimal? ActualRate { get; set; }
        public decimal TotalCashIn { get; set; }
        public decimal TotalCashOut { get; set; }
        public DateTime TransDate { get; set; }
        public decimal? CasinoTotalCashIn { get; set; }
        public decimal? CasinoTotalCashOut { get; set; }
        public DateTime? tstamp { get; set; }
    }
}
