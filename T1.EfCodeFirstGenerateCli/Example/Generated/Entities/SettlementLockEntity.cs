using System;

namespace Generated
{
    public class SettlementLockEntity
    {
        public bool IsLock { get; set; }
        public required string OperatorName { get; set; }
        public required string Action { get; set; }
        public required string Description { get; set; }
        public DateTime RequestDate { get; set; }
    }
}
