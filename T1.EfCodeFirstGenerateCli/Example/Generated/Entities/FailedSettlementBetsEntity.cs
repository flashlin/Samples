using System;

namespace Generated
{
    public class FailedSettlementBetsEntity
    {
        public long Id { get; set; }
        public long TransId { get; set; }
        public int MatchResultId { get; set; }
        public required string Action { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string Reason { get; set; }
        public required string CreatedBy { get; set; }
    }
}
