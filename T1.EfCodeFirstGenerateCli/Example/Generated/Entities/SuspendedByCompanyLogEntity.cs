using System;

namespace Generated
{
    public class SuspendedByCompanyLogEntity
    {
        public int SuspendedByCompanyLogId { get; set; }
        public int CustomerId { get; set; }
        public required string AccountId { get; set; }
        public bool? PreviousSuspended { get; set; }
        public bool PreviousSuspendedByCompany { get; set; }
        public required string SuspendedBy { get; set; }
        public DateTime SuspendedOn { get; set; }
    }
}
