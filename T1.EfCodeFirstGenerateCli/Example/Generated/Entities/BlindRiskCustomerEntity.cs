using System;

namespace Generated
{
    public class BlindRiskCustomerEntity
    {
        public int? CustId { get; set; }
        public bool? IsEnabled { get; set; }
        public bool? IsHighRiskPlayer { get; set; }
        public int? Score { get; set; }
        public decimal? BlindRiskRate { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public required string Remark { get; set; }
    }
}
