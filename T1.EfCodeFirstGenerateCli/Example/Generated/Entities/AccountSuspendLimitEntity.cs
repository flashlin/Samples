using System;

namespace Generated
{
    public class AccountSuspendLimitEntity
    {
        public int CustomerId { get; set; }
        public decimal SuspendLimit { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime ModifiedOn { get; set; }
        public bool IsEnabled { get; set; }
        public required string AccountId { get; set; }
    }
}
