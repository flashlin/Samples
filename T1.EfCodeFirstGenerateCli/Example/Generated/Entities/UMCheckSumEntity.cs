using System;

namespace Generated
{
    public class UMCheckSumEntity
    {
        public DateTime Winlostdate { get; set; }
        public required string CheckSumFloat { get; set; }
        public required string CheckType { get; set; }
        public decimal CheckSumDecimal { get; set; }
        public decimal? CheckTotalSumDecimal { get; set; }
        public DateTime? CreatedDate { get; set; }
        public decimal? CheckCasinoTotalSumDecimal { get; set; }
    }
}
