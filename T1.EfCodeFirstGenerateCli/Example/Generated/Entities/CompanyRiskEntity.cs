using System;

namespace Generated
{
    public class CompanyRiskEntity
    {
        public int CustID { get; set; }
        public required string UserName { get; set; }
        public decimal? Amount { get; set; }
        public required string LastModifiedBy { get; set; }
        public DateTime? LastModifiedOn { get; set; }
        public DateTime? tstamp { get; set; }
    }
}
