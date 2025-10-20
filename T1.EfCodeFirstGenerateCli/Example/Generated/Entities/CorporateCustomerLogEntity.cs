using System;

namespace Generated
{
    public class CorporateCustomerLogEntity
    {
        public int CustomerId { get; set; }
        public int CorporateGroupId { get; set; }
        public DateTime? StartDate { get; set; }
        public DateTime? EndDate { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime ModifiedOn { get; set; }
    }
}
